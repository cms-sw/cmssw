#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/ChordFinderSetter.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Application/interface/G4RegionReporter.h"
#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"
#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

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
#include "G4CascadeInterface.hh"

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RunManagerMT::RunManagerMT(edm::ParameterSet const & p):
      m_managerInitialized(false), 
      m_runTerminated(false),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_PhysicsTablesDir(p.getParameter<std::string>("PhysicsTablesDirectory")),
      m_StorePhysicsTables(p.getParameter<bool>("StorePhysicsTables")),
      m_RestorePhysicsTables(p.getParameter<bool>("RestorePhysicsTables")),
      m_pField(p.getParameter<edm::ParameterSet>("MagneticField")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),
      m_g4overlap(p.getParameter<edm::ParameterSet>("G4CheckOverlap")),
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_fieldBuilder(nullptr)
{    
  m_currentRun = nullptr;
  G4RunManagerKernel *kernel = G4MTRunManagerKernel::GetRunManagerKernel();
  if(!kernel) m_kernel = new G4MTRunManagerKernel();
  else {
    m_kernel = dynamic_cast<G4MTRunManagerKernel *>(kernel);
    assert(m_kernel);
  }

  m_check = p.getUntrackedParameter<bool>("CheckOverlap",false);
  m_WriteFile = p.getUntrackedParameter<std::string>("FileNameGDML","");
  m_FieldFile = p.getUntrackedParameter<std::string>("FileNameField","");
  m_RegionFile = p.getUntrackedParameter<std::string>("FileNameRegions","");
}

RunManagerMT::~RunManagerMT() 
{
  if(!m_runTerminated) { terminateRun(); }
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  G4GeometryManager::GetInstance()->OpenGeometry();
}

void RunManagerMT::initG4(const DDCompactView *pDD, const MagneticField *pMF, 
			  const HepPDT::ParticleDataTable *fPDGTable)
{
  if (m_managerInitialized) return;

  edm::LogInfo("SimG4CoreApplication") 
    << "RunManagerMT: start initialisation of geometry";
  
  // DDDWorld: get the DDCV from the ES and use it to build the World
  G4LogicalVolumeToDDLogicalPartMap map_;
  m_world.reset(new DDDWorld(pDD, map_, m_catalog, false));
  m_registry.dddWorldSignal_(m_world.get());

  // setup the magnetic field
  edm::LogInfo("SimG4CoreApplication") 
    << "RunManagerMT: start initialisation of magnetic field";

  if (m_pUseMagneticField)
    {
      const GlobalPoint g(0.,0.,0.);

      m_chordFinderSetter.reset(new sim::ChordFinderSetter());
      m_fieldBuilder = new sim::FieldBuilder(pMF, m_pField);
      G4TransportationManager * tM =
	G4TransportationManager::GetTransportationManager();
      m_fieldBuilder->build( tM->GetFieldManager(),
			     tM->GetPropagatorInField());
      if("" != m_FieldFile) {
	DumpMagneticField(tM->GetFieldManager()->GetDetectorField());
      }
    }

  // Create physics list
  edm::LogInfo("SimG4CoreApplication") 
    << "RunManagerMT: create PhysicsList";

  std::unique_ptr<PhysicsListMakerBase>
    physicsMaker(PhysicsListFactory::get()->create(
      m_pPhysics.getParameter<std::string> ("type")));
  if (physicsMaker.get()==0) {
    throw SimG4Exception("Unable to find the Physics list requested");
  }
  m_physicsList = 
    physicsMaker->make(map_,fPDGTable,m_chordFinderSetter.get(),m_pPhysics,m_registry);

  PhysicsList* phys = m_physicsList.get(); 
  if (phys==0) { 
    throw SimG4Exception("Physics list construction failed!"); 
  }

  // adding GFlash, Russian Roulette for eletrons and gamma, 
  // step limiters on top of any Physics Lists
  phys->RegisterPhysics(new ParametrisedEMPhysics("EMoptions",m_pPhysics));

  m_physicsList->ResetStoredInAscii();
  if (m_RestorePhysicsTables) {
    m_physicsList->SetPhysicsTableRetrieved(m_PhysicsTablesDir);
  }
  edm::LogInfo("SimG4CoreApplication") 
    << "RunManagerMT: start initialisation of PhysicsList for master";
  
  m_kernel->SetPhysics(phys);
  m_kernel->InitializePhysics();
  m_kernel->SetUpDecayChannels();

  // The following line was with the following comment in
  // G4MTRunManager::InitializePhysics() in 10.00.p01; in practice
  // needed to initialize certain singletons during the master thread
  // initialization in order to avoid races later...
  //
  //BERTINI, this is needed to create pseudo-particles, to be removed
  G4CascadeInterface::Initialize();

  if (m_kernel->RunInitialization()) { m_managerInitialized = true; }
  else { 
    throw SimG4Exception("G4RunManagerKernel initialization failed!"); 
  }

  if (m_StorePhysicsTables) {
    std::ostringstream dir;
    dir << m_PhysicsTablesDir << '\0';
    std::string cmd = std::string("/control/shell mkdir -p ")+m_PhysicsTablesDir;
    if (!std::ifstream(dir.str().c_str(), std::ios::in))
      G4UImanager::GetUIpointer()->ApplyCommand(cmd);
    m_physicsList->StorePhysicsTable(m_PhysicsTablesDir);
  }

  initializeUserActions();

  if(0 < m_G4Commands.size()) {
    G4cout << "RunManagerMT: Requested UI commands: " << G4endl;
    for (unsigned it=0; it<m_G4Commands.size(); ++it) {
      G4cout << "    " << m_G4Commands[it] << G4endl;
      G4UImanager::GetUIpointer()->ApplyCommand(m_G4Commands[it]);
    }
  }

  // geometry dump
  if("" != m_WriteFile) {
    G4GDMLParser gdml(new G4GDMLReadStructure(), new CMSGDMLWriteStructure());
    gdml.Write(m_WriteFile, m_world->GetWorldVolume(), true);
  }

  // G4Region dump
  if("" != m_RegionFile) {
    G4RegionReporter rrep;
    rrep.ReportRegions(m_RegionFile);
  }

  // Intersection check
  if(m_check) { G4CheckOverlap check(m_g4overlap); }

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //G4ParticleTable::GetParticleTable()->DumpTable("ALL");
  //
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  m_currentRun = new G4Run(); 
  m_userRunAction->BeginOfRunAction(m_currentRun); 
}

void RunManagerMT::initializeUserActions() {
  m_runInterface.reset(new SimRunInterface(this, true));
  m_userRunAction = new RunAction(m_pRunAction, m_runInterface.get());
  Connect(m_userRunAction);
}

void  RunManagerMT::Connect(RunAction* runAction)
{
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void RunManagerMT::stopG4()
{
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  if(!m_runTerminated) { terminateRun(); }
}

void RunManagerMT::terminateRun() {
  m_userRunAction->EndOfRunAction(m_currentRun);
  delete m_userRunAction;
  m_userRunAction = 0;
  if(m_kernel && !m_runTerminated) {
    m_kernel->RunTermination();
    m_runTerminated = true;
  }
}

void RunManagerMT::DumpMagneticField(const G4Field* field) const
{
  std::ofstream fout(m_FieldFile.c_str(), std::ios::out);
  if(fout.fail()){
    edm::LogWarning("SimG4CoreApplication") 
      << " RunManager WARNING : "
      << "error opening file <" << m_FieldFile << "> for magnetic field";
  } else {
    double rmax = 9000*mm;
    double zmax = 16000*mm;

    double dr = 5*cm;
    double dz = 20*cm;

    int nr = (int)(rmax/dr);
    int nz = 2*(int)(zmax/dz);

    double r = 0.0;
    double z0 = -zmax;
    double z;

    double phi = 0.0;
    double cosf = cos(phi);
    double sinf = sin(phi);

    double point[4] = {0.0,0.0,0.0,0.0};
    double bfield[3] = {0.0,0.0,0.0};

    fout << std::setprecision(6);
    for(int i=0; i<=nr; ++i) {
      z = z0;
      for(int j=0; j<=nz; ++j) {
        point[0] = r*cosf;
	point[1] = r*sinf;
	point[2] = z;
        field->GetFieldValue(point, bfield);
        fout << "R(mm)= " << r/mm << " phi(deg)= " << phi/degree
	     << " Z(mm)= " << z/mm << "   Bz(tesla)= " << bfield[2]/tesla
	     << " Br(tesla)= " << (bfield[0]*cosf + bfield[1]*sinf)/tesla
	     << " Bphi(tesla)= " << (bfield[0]*sinf - bfield[1]*cosf)/tesla
	     << G4endl;
	z += dz;
      }
      r += dr;
    }

    fout.close();
  }
}
