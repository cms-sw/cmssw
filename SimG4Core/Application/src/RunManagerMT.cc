#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"
#include "SimG4Core/Application/interface/ExceptionHandler.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/DDG4ProductionCuts.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/CustomPhysics/interface/CMSExoticaPhysics.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Application/interface/G4RegionReporter.h"
#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"
#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

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
#include "G4CascadeInterface.hh"

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"

#include "DDG4/Geant4Mapping.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RunManagerMT::RunManagerMT(edm::ParameterSet const& p)
    : m_managerInitialized(false),
      m_runTerminated(false),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_PhysicsTablesDir(p.getParameter<std::string>("PhysicsTablesDirectory")),
      m_StorePhysicsTables(p.getParameter<bool>("StorePhysicsTables")),
      m_RestorePhysicsTables(p.getParameter<bool>("RestorePhysicsTables")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),
      m_g4overlap(p.getParameter<edm::ParameterSet>("G4CheckOverlap")),
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_p(p) {
  m_currentRun = nullptr;
  m_UIsession.reset(new CustomUIsession());
  m_physicsList.reset(nullptr);
  m_world.reset(nullptr);

  m_runInterface.reset(nullptr);
  m_userRunAction = nullptr;
  m_currentRun = nullptr;

  m_kernel = new G4MTRunManagerKernel();
  m_stateManager = G4StateManager::GetStateManager();
  m_stateManager->SetExceptionHandler(new ExceptionHandler());
  m_geometryManager->G4GeometryManager::GetInstance();

  m_check = p.getUntrackedParameter<bool>("CheckOverlap", false);
}

RunManagerMT::~RunManagerMT() { stopG4(); }

void RunManagerMT::initG4(const DDCompactView* pDD,
                          const cms::DDCompactView* pDD4hep,
                          const MagneticField* pMF,
                          const HepPDT::ParticleDataTable* fPDGTable) {
  edm::LogWarning("SimG4CoreApplication") << "### RunManagerMT::initG4";
  if (m_managerInitialized) {
    edm::LogWarning("SimG4CoreApplication") << "RunManagerMT::initG4 was already done - exit";
    return;
  }
  auto geoFromDD4hep = m_p.getParameter<bool>("g4GeometryDD4hepSource");
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: start initialising of geometry DD4Hep: " << geoFromDD4hep;

  // DDDWorld: get the DDCV from the ES and use it to build the World
  G4LogicalVolumeToDDLogicalPartMap map_lv;
  dd4hep::sim::Geant4GeometryMaps::VolumeMap lvMap;
  if (geoFromDD4hep) {
    m_world.reset(new DDDWorld(pDD4hep->detector(), lvMap));
  } else {
    m_world.reset(new DDDWorld(pDD, map_lv, m_catalog, false));
  }
  G4VPhysicalVolume* world = m_world.get()->GetWorldVolume();

  bool cuts = m_pPhysics.getParameter<bool>("CutsPerRegion");
  int verb =
      std::max(m_pPhysics.getUntrackedParameter<int>("Verbosity", 0), m_p.getParameter<int>("SteppingVerbosity"));
  m_kernel->SetVerboseLevel(verb);
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: Define cuts: " << cuts << " Geant4 run manager verbosity: " << verb;

  if (cuts) {
    DDG4ProductionCuts pcuts(map_lv, verb, m_pPhysics);
  }
  const G4RegionStore* regStore = G4RegionStore::GetInstance();
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();
  unsigned int numLV = lvs->size();
  unsigned int nn = regStore->size();
  edm::LogVerbatim("SimG4CoreApplication")
      << "###RunManagerMT: " << numPV << " PhysVolumes; " << numLV << " LogVolumes; " << nn << " Regions.";

  m_kernel->DefineWorldVolume(world, true);
  m_registry.dddWorldSignal_(m_world.get());

  // Create physics list
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: create PhysicsList";

  std::unique_ptr<PhysicsListMakerBase> physicsMaker(
      PhysicsListFactory::get()->create(m_pPhysics.getParameter<std::string>("type")));
  if (physicsMaker.get() == nullptr) {
    throw edm::Exception(edm::errors::Configuration) << "Unable to find the Physics list requested";
  }
  m_physicsList = physicsMaker->make(m_pPhysics, m_registry);

  PhysicsList* phys = m_physicsList.get();
  if (phys == nullptr) {
    throw edm::Exception(edm::errors::Configuration, "Physics list construction failed!");
  }

  // exotic particle physics
  double monopoleMass = m_pPhysics.getUntrackedParameter<double>("MonopoleMass", 0);
  if (monopoleMass > 0.0) {
    phys->RegisterPhysics(new CMSMonopolePhysics(fPDGTable, m_pPhysics));
  }
  bool exotica = m_pPhysics.getUntrackedParameter<bool>("ExoticaTransport", false);
  if (exotica) {
    CMSExoticaPhysics exo(phys, m_pPhysics);
  }

  // adding GFlash, Russian Roulette for eletrons and gamma,
  // step limiters on top of any Physics Lists
  phys->RegisterPhysics(new ParametrisedEMPhysics("EMoptions", m_pPhysics));

  if (m_RestorePhysicsTables) {
    m_physicsList->SetPhysicsTableRetrieved(m_PhysicsTablesDir);
  }
  edm::LogInfo("SimG4CoreApplication") << "RunManagerMT: start initialisation of PhysicsList for master";

  if (!cuts) {
    m_physicsList->SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue") * CLHEP::cm);
    m_physicsList->SetCutsWithDefault();
  }

  m_kernel->SetPhysics(phys);

  edm::LogInfo("SimG4CoreApplication") << "RunManagerMT: PhysicsList and cuts are defined";

  // Geant4 UI commands before initialisation of physics
  if (!m_G4Commands.empty()) {
    G4cout << "RunManagerMT: Requested UI commands: " << G4endl;
    for (const std::string& command : m_G4Commands) {
      G4cout << "    " << command << G4endl;
      G4UImanager::GetUIpointer()->ApplyCommand(command);
    }
  }

  m_stateManager->SetNewState(G4State_Init);
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: G4State is Init";
  m_kernel->InitializePhysics();
  m_kernel->SetUpDecayChannels();

  // The following line was with the following comment in
  // G4MTRunManager::InitializePhysics() in 10.00.p01; in practice
  // needed to initialize certain singletons during the master thread
  // initialization in order to avoid races later...
  //
  //BERTINI, this is needed to create pseudo-particles, to be removed
  G4CascadeInterface::Initialize();

  if (m_kernel->RunInitialization()) {
    m_managerInitialized = true;
  } else {
    throw edm::Exception(edm::errors::LogicError, "G4RunManagerKernel initialization failed!");
  }

  if (m_StorePhysicsTables) {
    std::ostringstream dir;
    dir << m_PhysicsTablesDir << '\0';
    std::string cmd = std::string("/control/shell mkdir -p ") + m_PhysicsTablesDir;
    if (!std::ifstream(dir.str().c_str(), std::ios::in))
      G4UImanager::GetUIpointer()->ApplyCommand(cmd);
    m_physicsList->StorePhysicsTable(m_PhysicsTablesDir);
  }

  if (verb > 1) {
    m_physicsList->DumpCutValuesTable();
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: Physics is initilized, now initialise user actions";

  initializeUserActions();

  // geometry dump
  auto writeFile = m_p.getUntrackedParameter<std::string>("FileNameGDML", "");
  if (!writeFile.empty()) {
    G4GDMLParser gdml;
    gdml.SetRegionExport(true);
    gdml.SetEnergyCutsExport(true);
    gdml.Write(writeFile, m_world->GetWorldVolume(), true);
  }

  // G4Region dump
  auto regionFile = m_p.getUntrackedParameter<std::string>("FileNameRegions", "");
  if (!regionFile.empty()) {
    G4RegionReporter rrep;
    rrep.ReportRegions(regionFile);
  }

  // Intersection check
  if (m_check) {
    G4CheckOverlap check(m_g4overlap);
  }

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //G4ParticleTable::GetParticleTable()->DumpTable("ALL");
  //
  m_stateManager->SetNewState(G4State_GeomClosed);
  m_currentRun = new G4Run();
  m_userRunAction->BeginOfRunAction(m_currentRun);
}

void RunManagerMT::initializeUserActions() {
  m_runInterface.reset(new SimRunInterface(this, true));
  m_userRunAction = new RunAction(m_pRunAction, m_runInterface.get(), true);
  Connect(m_userRunAction);
}

void RunManagerMT::Connect(RunAction* runAction) {
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void RunManagerMT::stopG4() {
  m_geometryManager->OpenGeometry();
  m_stateManager->SetNewState(G4State_Quit);
  if (!m_runTerminated) {
    terminateRun();
  }
}

void RunManagerMT::terminateRun() {
  if (m_userRunAction) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    delete m_userRunAction;
    m_userRunAction = nullptr;
  }
  if (m_kernel && !m_runTerminated) {
    m_kernel->RunTermination();
  }
  m_runTerminated = true;
}
