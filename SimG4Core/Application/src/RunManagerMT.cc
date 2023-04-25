#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/ExceptionHandler.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/CustomUIsession.h"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/CustomPhysics/interface/CMSExoticaPhysics.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"
#include "SimG4Core/Geometry/interface/CMSG4CheckOverlap.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "HepPDT/ParticleDataTable.hh"

#include "G4Timer.hh"
#include "G4GeometryManager.hh"
#include "G4ScoringManager.hh"
#include "G4StateManager.hh"
#include "G4ApplicationState.hh"
#include "G4MTRunManagerKernel.hh"
#include "G4UImanager.hh"

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"
#include "G4ParticleTable.hh"
#include "G4CascadeInterface.hh"
#include "G4EmParameters.hh"
#include "G4HadronicParameters.hh"
#include "G4NuclearLevelData.hh"

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"

#include <iostream>
#include <memory>

#include <sstream>
#include <fstream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

RunManagerMT::RunManagerMT(edm::ParameterSet const& p)
    : m_managerInitialized(false),
      m_runTerminated(false),
      m_PhysicsTablesDir(p.getUntrackedParameter<std::string>("PhysicsTablesDirectory", "")),
      m_StorePhysicsTables(p.getUntrackedParameter<bool>("StorePhysicsTables", false)),
      m_RestorePhysicsTables(p.getUntrackedParameter<bool>("RestorePhysicsTables", false)),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),
      m_g4overlap(p.getUntrackedParameter<edm::ParameterSet>("G4CheckOverlap")),
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_p(p) {
  m_currentRun = nullptr;
  m_UIsession = new CustomUIsession();
  m_physicsList.reset(nullptr);
  m_world.reset(nullptr);

  m_runInterface.reset(nullptr);
  m_userRunAction = nullptr;
  m_currentRun = nullptr;

  m_kernel = new G4MTRunManagerKernel();
  m_stateManager = G4StateManager::GetStateManager();
  double th = p.getParameter<double>("ThresholdForGeometryExceptions") * CLHEP::GeV;
  bool tr = p.getParameter<bool>("TraceExceptions");
  m_stateManager->SetExceptionHandler(new ExceptionHandler(th, tr));
  m_check = p.getUntrackedParameter<bool>("CheckGeometry", false);
}

RunManagerMT::~RunManagerMT() { delete m_UIsession; }

void RunManagerMT::initG4(const DDCompactView* pDD,
                          const cms::DDCompactView* pDD4hep,
                          const HepPDT::ParticleDataTable* fPDGTable) {
  if (m_managerInitialized) {
    edm::LogWarning("SimG4CoreApplication") << "RunManagerMT::initG4 was already done - exit";
    return;
  }
  bool geoFromDD4hep = m_p.getParameter<bool>("g4GeometryDD4hepSource");
  bool cuts = m_pPhysics.getParameter<bool>("CutsPerRegion");
  bool protonCut = m_pPhysics.getParameter<bool>("CutsOnProton");
  int verb = m_pPhysics.getUntrackedParameter<int>("Verbosity", 0);
  int stepverb = m_p.getUntrackedParameter<int>("SteppingVerbosity", 0);
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: start initialising of geometry DD4hep: " << geoFromDD4hep << "\n"
      << "              cutsPerRegion: " << cuts << " cutForProton: " << protonCut << "\n"
      << "              G4 verbosity: " << verb;

  G4Timer timer;
  timer.Start();

  G4UImanager::GetUIpointer()->SetCoutDestination(m_UIsession);
  G4UImanager::GetUIpointer()->SetMasterUIManager(true);

  m_world = std::make_unique<DDDWorld>(pDD, pDD4hep, m_catalog, verb, cuts, protonCut);
  G4VPhysicalVolume* world = m_world.get()->GetWorldVolume();

  m_kernel->SetVerboseLevel(verb);
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: Define cuts: " << cuts << " Geant4 run manager verbosity: " << verb;

  const G4RegionStore* regStore = G4RegionStore::GetInstance();
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();
  unsigned int numLV = lvs->size();
  unsigned int nn = regStore->size();
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: " << numPV << " physical volumes; " << numLV << " logical volumes; " << nn << " regions.";

  if (m_check) {
    m_kernel->SetVerboseLevel(2);
  }
  m_kernel->DefineWorldVolume(world, true);
  m_registry.dddWorldSignal_(m_world.get());
  m_stateManager->SetNewState(G4State_PreInit);

  // Create physics list
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: create PhysicsList";

  std::unique_ptr<PhysicsListMakerBase> physicsMaker(
      PhysicsListFactory::get()->create(m_pPhysics.getParameter<std::string>("type")));
  if (physicsMaker.get() == nullptr) {
    throw cms::Exception("Configuration") << "Unable to find the Physics list requested";
  }
  m_physicsList = physicsMaker->make(m_pPhysics, m_registry);

  PhysicsList* phys = m_physicsList.get();
  if (phys == nullptr) {
    throw cms::Exception("Configuration") << "Physics list construction failed!";
  }
  if (stepverb > 0) {
    verb = std::max(verb, 1);
  }
  G4HadronicParameters::Instance()->SetVerboseLevel(verb);
  G4EmParameters::Instance()->SetVerbose(verb);
  G4EmParameters::Instance()->SetWorkerVerbose(std::max(verb - 1, 0));

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
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: start initialisation of PhysicsList for master";

  m_physicsList->SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue") * CLHEP::cm);
  m_physicsList->SetCutsWithDefault();
  m_kernel->SetPhysics(phys);

  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: PhysicsList and cuts are defined";

  // Enable couple transportation
  bool scorer = m_p.getParameter<bool>("UseCommandBaseScorer");
  if (scorer) {
    G4ScoringManager* scManager = G4ScoringManager::GetScoringManager();
    scManager->SetVerboseLevel(1);
  }
  // Geant4 UI commands before initialisation of physics
  if (!m_G4Commands.empty()) {
    edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: Requested UI commands: ";
    for (const std::string& command : m_G4Commands) {
      edm::LogVerbatim("SimG4CoreApplication") << "    " << command;
      G4UImanager::GetUIpointer()->ApplyCommand(command);
    }
  }

  setupVoxels();

  m_stateManager->SetNewState(G4State_Init);
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: G4State is Init";
  m_kernel->InitializePhysics();
  if (verb > 0) {
    G4EmParameters::Instance()->Dump();
  }
  m_kernel->SetUpDecayChannels();

  if (m_kernel->RunInitialization()) {
    m_managerInitialized = true;
  } else {
    throw cms::Exception("LogicError") << "G4RunManagerKernel initialization failed!";
  }

  if (m_check) {
    checkVoxels();
  }

  if (m_StorePhysicsTables) {
    std::ostringstream dir;
    dir << m_PhysicsTablesDir << '\0';
    std::string cmd = std::string("/control/shell mkdir -p ") + m_PhysicsTablesDir;
    if (!std::ifstream(dir.str().c_str(), std::ios::in))
      G4UImanager::GetUIpointer()->ApplyCommand(cmd);
    m_physicsList->StorePhysicsTable(m_PhysicsTablesDir);
  }
  // Appload nuclear level data up to Z=84
  G4NuclearLevelData::GetInstance()->UploadNuclearLevelData(84);

  if (verb > 1) {
    m_physicsList->DumpCutValuesTable();
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: Physics is initilized, now initialise user actions, verb=" << verb;

  initializeUserActions();

  // geometry dump
  auto writeFile = m_p.getUntrackedParameter<std::string>("FileNameGDML", "");
  if (!writeFile.empty()) {
    G4GDMLParser gdml;
    gdml.SetRegionExport(true);
    gdml.SetEnergyCutsExport(true);
    gdml.Write(writeFile, m_world->GetWorldVolume(), true);
  }

  // G4Region dump file name
  auto regionFile = m_p.getUntrackedParameter<std::string>("FileNameRegions", "");
  runForPhase2();  

  // Geometry checks
  if (m_check || !regionFile.empty()) {
    CMSG4CheckOverlap check(m_g4overlap, regionFile, m_UIsession, world);
  }

  m_stateManager->SetNewState(G4State_PreInit);
  G4HadronicParameters::Instance()->SetVerboseLevel(std::max(verb - 1, 0));

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //G4ParticleTable::GetParticleTable()->DumpTable("ALL");
  //
  m_stateManager->SetNewState(G4State_GeomClosed);
  m_currentRun = new G4Run();
  m_userRunAction->BeginOfRunAction(m_currentRun);
  timer.Stop();
  G4cout.precision(4);
  G4cout << "RunManagerMT: initG4 done " << timer << G4endl;
}

void RunManagerMT::initializeUserActions() {
  m_runInterface = std::make_unique<SimRunInterface>(this, true);
  m_userRunAction = new RunAction(m_pRunAction, m_runInterface.get(), true);
  Connect(m_userRunAction);
}

void RunManagerMT::Connect(RunAction* runAction) {
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void RunManagerMT::stopG4() {
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT::stopG4";
  G4GeometryManager::GetInstance()->OpenGeometry();
  m_stateManager->SetNewState(G4State_Quit);
  if (!m_runTerminated) {
    terminateRun();
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT::stopG4 done";
}

void RunManagerMT::terminateRun() {
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT::terminateRun";
  if (nullptr != m_userRunAction) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    delete m_userRunAction;
    m_userRunAction = nullptr;
  }
  if (!m_runTerminated) {
    m_kernel->RunTermination();
  }
  m_runTerminated = true;
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT::terminateRun done";
}

void RunManagerMT::checkVoxels() {
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  int numLV = lvs->size();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMT: nLV=" << numLV;
  int nvox = 0;
  int nslice = 0;
  for (int i = 0; i < numLV; ++i) {
    auto lv = (*lvs)[i];
    auto nd = lv->GetNoDaughters();
    auto vox = lv->GetVoxelHeader();
    auto sma = lv->GetSmartless();
    auto reg = lv->GetRegion();
    size_t nsl = (nullptr == vox) ? 0 : vox->GetNoSlices();
    if (0 < nsl) {
      nslice += nsl;
      std::string rname = (nullptr != reg) ? reg->GetName() : "";
      edm::LogVerbatim("Voxels") << " " << i << ". Nd=" << nd << " Nsl=" << nsl << " Smartless=" << sma << " "
                                 << lv->GetName() << " Region: " << rname;
      ++nvox;
    }
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: nLV=" << numLV << " NlvVox=" << nvox << " Nslices=" << nslice;
}

void RunManagerMT::setupVoxels() {
  double density = m_p.getParameter<double>("DefaultVoxelDensity");
  std::vector<std::string> rnames = m_p.getParameter<std::vector<std::string> >("VoxelRegions");
  std::vector<double> rdensities = m_p.getParameter<std::vector<double> >("VoxelDensityPerRegion");
  int nr = 0;
  std::size_t n = rnames.size();
  if (n == rdensities.size()) {
    nr = (int)n;
  }
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  for (auto& lv : *lvs) {
    double den = density;
    if (0 < nr) {
      std::string nam = lv->GetRegion()->GetName();
      for (int i = 0; i < nr; ++i) {
        if (nam == rnames[i]) {
          den = rdensities[i];
          break;
        }
      }
    }
    lv->SetSmartless(den);
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMT: default voxel density=" << density << "; number of regions with special density " << nr;
}

void RunManagerMT::runForPhase2() {
  const G4RegionStore* regStore = G4RegionStore::GetInstance();
  for(auto & r : *regStore) {
    const G4String& name = r->GetName();
    if(name == "HGCalRegion" ||
       name == "FastTimerRegionETL" ||
       name == "FastTimerRegionBTL") {
      m_isPhase2 = true;
      break;
    }
  }
}
