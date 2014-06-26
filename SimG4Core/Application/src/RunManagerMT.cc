#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "G4GeometryManager.hh"
#include "G4StateManager.hh"
#include "G4ApplicationState.hh"
#include "G4MTRunManagerKernel.hh"
#include "G4UImanager.hh"

#include "G4EventManager.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"
#include "G4ParticleTable.hh"
#include "G4Field.hh"
#include "G4FieldManager.hh"

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "SimG4Core/Application/interface/ExceptionHandler.h"

RunManagerMT::RunManagerMT(edm::ParameterSet const & p) 
  :   m_generator(0), m_nonBeam(p.getParameter<bool>("NonBeamEvent")), 
      m_primaryTransformer(0), 
      m_managerInitialized(false), 
      m_runInitialized(false), m_runTerminated(false), m_runAborted(false),
      firstRun(true),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_PhysicsTablesDir(p.getParameter<std::string>("PhysicsTablesDirectory")),
      m_StorePhysicsTables(p.getParameter<bool>("StorePhysicsTables")),
      m_RestorePhysicsTables(p.getParameter<bool>("RestorePhysicsTables")),
      m_pField(p.getParameter<edm::ParameterSet>("MagneticField")),
      m_pGenerator(p.getParameter<edm::ParameterSet>("Generator")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),      
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_p(p), m_fieldBuilder(0),
      m_theLHCTlinkTag(p.getParameter<edm::InputTag>("theLHCTlinkTag"))
{    
  //m_HepMC = iC.consumes<edm::HepMCProduct>(p.getParameter<edm::InputTag>("HepMCProduct"));

  G4RunManagerKernel *kernel = G4MTRunManagerKernel::GetRunManagerKernel();
  if(!kernel) m_kernel = new G4MTRunManagerKernel();
  else {
    m_kernel = dynamic_cast<G4MTRunManagerKernel *>(kernel);
    assert(m_kernel);
  }

  m_check = p.getUntrackedParameter<bool>("CheckOverlap",false);
  m_WriteFile = p.getUntrackedParameter<std::string>("FileNameGDML","");
  m_FieldFile = p.getUntrackedParameter<std::string>("FileNameField","");
  if("" != m_FieldFile) { m_FieldFile += ".txt"; } 

  m_InTag = m_pGenerator.getParameter<std::string>("HepMCProductLabel") ;

}

RunManagerMT::~RunManagerMT() 
{ 
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  G4GeometryManager::GetInstance()->OpenGeometry();
}

void RunManagerMT::initG4(const DDCompactView *pDD, const MagneticField *pMF, const HepPDT::ParticleDataTable *fPDGTable, const edm::EventSetup & es)
{
  if (m_managerInitialized) return;
  
  // DDDWorld: get the DDCV from the ES and use it to build the World
  G4LogicalVolumeToDDLogicalPartMap map_;
  SensitiveDetectorCatalog catalog_;
  m_world.reset(new DDDWorld(pDD, map_, catalog_, m_check));
  m_registry.dddWorldSignal_(m_world.get());

  std::auto_ptr<PhysicsListMakerBase> 
    physicsMaker(PhysicsListFactory::get()->create(
      m_pPhysics.getParameter<std::string> ("type")));
  if (physicsMaker.get()==0) {
    throw SimG4Exception("Unable to find the Physics list requested");
  }
  m_physicsList = 
    physicsMaker->make(map_,fPDGTable,m_fieldBuilder,m_pPhysics,m_registry);

  PhysicsList* phys = m_physicsList.get(); 
  if (phys==0) { 
    throw SimG4Exception("Physics list construction failed!"); 
  }

  // adding GFlash, Russian Roulette for eletrons and gamma, 
  // step limiters on top of any Physics Lists
  phys->RegisterPhysics(new ParametrisedEMPhysics("EMoptions",m_pPhysics));
  
  m_kernel->SetPhysics(phys);
  m_kernel->InitializePhysics();

  if (m_kernel->RunInitialization()) { m_managerInitialized = true; }
  else { 
    throw SimG4Exception("G4RunManagerKernel initialization failed!"); 
  }

  initializeUserActions();

  for (unsigned it=0; it<m_G4Commands.size(); it++) {
    edm::LogInfo("SimG4CoreApplication") << "RunManagerMT:: Requests UI: "
                                         << m_G4Commands[it];
    G4UImanager::GetUIpointer()->ApplyCommand(m_G4Commands[it]);
  }

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //  G4cout << "Output of G4ParticleTable DumpTable:" << G4endl;
  //  G4ParticleTable::GetParticleTable()->DumpTable("ALL");
  
  firstRun= false;
}

void RunManagerMT::initializeUserActions() {
  m_runInterface.reset(new SimRunInterface(this, true));

  m_userRunAction.reset(new RunAction(m_pRunAction, m_runInterface.get()));
  Connect(m_userRunAction.get());
}

void  RunManagerMT::Connect(RunAction* runAction)
{
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void RunManagerMT::stopG4()
{
  //std::cout << "RunManagerMT::stopG4" << std::endl;
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
}

