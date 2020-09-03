#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"
#include "SimG4Core/Application/interface/ExceptionHandler.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/CustomPhysics/interface/CMSExoticaPhysics.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "SimG4Core/Geometry/interface/CustomUIsession.h"
#include "SimG4Core/Geometry/interface/CMSG4CheckOverlap.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

#include "G4GeometryManager.hh"
#include "G4StateManager.hh"
#include "G4ApplicationState.hh"
#include "G4RunManagerKernel.hh"
#include "G4UImanager.hh"

#include "G4EventManager.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4TransportationManager.hh"
#include "G4ParticleTable.hh"
#include "G4Field.hh"
#include "G4FieldManager.hh"

#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <memory>

#include <fstream>
#include <memory>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

static void createWatchers(const edm::ParameterSet& iP,
                           SimActivityRegistry& iReg,
                           std::vector<std::shared_ptr<SimWatcher> >& oWatchers,
                           std::vector<std::shared_ptr<SimProducer> >& oProds) {
  using namespace std;
  using namespace edm;

  vector<ParameterSet> watchers = iP.getParameter<vector<ParameterSet> >("Watchers");

  for (vector<ParameterSet>::iterator itWatcher = watchers.begin(); itWatcher != watchers.end(); ++itWatcher) {
    std::shared_ptr<SimWatcherMakerBase> maker(
        SimWatcherFactory::get()->create(itWatcher->getParameter<std::string>("type")));
    if (maker.get() == nullptr) {
      throw edm::Exception(edm::errors::Configuration) << "Unable to find the requested Watcher";
    }

    std::shared_ptr<SimWatcher> watcherTemp;
    std::shared_ptr<SimProducer> producerTemp;
    maker->make(*itWatcher, iReg, watcherTemp, producerTemp);
    oWatchers.push_back(watcherTemp);
    if (producerTemp) {
      oProds.push_back(producerTemp);
    }
  }
}

RunManager::RunManager(edm::ParameterSet const& p, edm::ConsumesCollector&& iC)
    : m_generator(new Generator(p.getParameter<edm::ParameterSet>("Generator"))),
      m_HepMC(iC.consumes<edm::HepMCProduct>(
          p.getParameter<edm::ParameterSet>("Generator").getParameter<edm::InputTag>("HepMCProductLabel"))),
      m_LHCtr(iC.consumes<edm::LHCTransportLinkContainer>(p.getParameter<edm::InputTag>("theLHCTlinkTag"))),
      m_nonBeam(p.getParameter<bool>("NonBeamEvent")),
      m_primaryTransformer(nullptr),
      m_managerInitialized(false),
      m_runInitialized(false),
      m_runTerminated(false),
      m_runAborted(false),
      firstRun(true),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_currentRun(nullptr),
      m_currentEvent(nullptr),
      m_simEvent(nullptr),
      m_PhysicsTablesDir(p.getUntrackedParameter<std::string>("PhysicsTablesDirectory", "")),
      m_StorePhysicsTables(p.getUntrackedParameter<bool>("StorePhysicsTables", false)),
      m_RestorePhysicsTables(p.getUntrackedParameter<bool>("RestorePhysicsTables", false)),
      m_UseParametrisedEMPhysics(p.getUntrackedParameter<bool>("UseParametrisedEMPhysics")),
      m_EvtMgrVerbosity(p.getUntrackedParameter<int>("G4EventManagerVerbosity", 0)),
      m_pField(p.getParameter<edm::ParameterSet>("MagneticField")),
      m_pGenerator(p.getParameter<edm::ParameterSet>("Generator")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),
      m_pEventAction(p.getParameter<edm::ParameterSet>("EventAction")),
      m_pStackingAction(p.getParameter<edm::ParameterSet>("StackingAction")),
      m_pTrackingAction(p.getParameter<edm::ParameterSet>("TrackingAction")),
      m_pSteppingAction(p.getParameter<edm::ParameterSet>("SteppingAction")),
      m_g4overlap(p.getUntrackedParameter<edm::ParameterSet>("G4CheckOverlap")),
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_p(p) {
  m_UIsession = new CustomUIsession();
  m_kernel = new G4RunManagerKernel();
  G4StateManager::GetStateManager()->SetExceptionHandler(new ExceptionHandler());

  m_physicsList.reset(nullptr);

  m_check = p.getUntrackedParameter<bool>("CheckGeometry", false);
  m_WriteFile = p.getUntrackedParameter<std::string>("FileNameGDML", "");
  m_FieldFile = p.getUntrackedParameter<std::string>("FileNameField", "");
  m_RegionFile = p.getUntrackedParameter<std::string>("FileNameRegions", "");

  m_userRunAction = nullptr;
  m_runInterface = nullptr;

  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  edm::Service<SimActivityRegistry> otherRegistry;
  if (otherRegistry) {
    m_registry.connect(*otherRegistry);
  }
  m_sVerbose.reset(nullptr);

  std::vector<edm::ParameterSet> watchers = p.getParameter<std::vector<edm::ParameterSet> >("Watchers");
  m_hasWatchers = (watchers.empty()) ? false : true;

  if (m_hasWatchers) {
    createWatchers(m_p, m_registry, m_watchers, m_producers);
  }
}

RunManager::~RunManager() {
  if (!m_runTerminated) {
    terminateRun();
  }
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  G4GeometryManager::GetInstance()->OpenGeometry();
  //   if (m_kernel!=0) delete m_kernel;
  delete m_runInterface;
  delete m_generator;
}

void RunManager::initG4(const edm::EventSetup& es) {
  bool geomChanged = idealGeomRcdWatcher_.check(es);
  if (geomChanged && (!firstRun)) {
    throw cms::Exception("BadConfig") << "[SimG4Core RunManager]\n"
                                      << "The Geometry configuration is changed during the job execution\n"
                                      << "this is not allowed, the geometry must stay unchanged\n";
  }
  bool geoFromDD4hep = m_p.getParameter<bool>("g4GeometryDD4hepSource");
  bool cuts = m_pPhysics.getParameter<bool>("CutsPerRegion");
  bool protonCut = m_pPhysics.getParameter<bool>("CutsOnProton");
  int verb = std::max(m_pPhysics.getUntrackedParameter<int>("Verbosity", 0),
                      m_p.getUntrackedParameter<int>("SteppingVerbosity", 0));
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManager: start initialising of geometry DD4Hep: " << geoFromDD4hep << "\n"
      << "              cutsPerRegion: " << cuts << " cutForProton: " << protonCut << "\n"
      << "              G4 verbosity: " << verb;

  if (m_pUseMagneticField) {
    bool magChanged = idealMagRcdWatcher_.check(es);
    if (magChanged && (!firstRun)) {
      throw edm::Exception(edm::errors::Configuration)
          << "[SimG4Core RunManager]\n"
          << "The MagneticField configuration is changed during the job execution\n"
          << "this is not allowed, the MagneticField must stay unchanged\n";
    }
  }

  if (m_managerInitialized)
    return;

  // initialise geometry
  const DDCompactView* pDD = nullptr;
  const cms::DDCompactView* pDD4hep = nullptr;
  if (geoFromDD4hep) {
    edm::ESTransientHandle<cms::DDCompactView> ph;
    es.get<IdealGeometryRecord>().get(ph);
    pDD4hep = ph.product();
  } else {
    edm::ESTransientHandle<DDCompactView> ph;
    es.get<IdealGeometryRecord>().get(ph);
    pDD = ph.product();
  }
  SensitiveDetectorCatalog catalog;
  const DDDWorld* world = new DDDWorld(pDD, pDD4hep, catalog, verb, cuts, protonCut);
  G4VPhysicalVolume* pworld = world->GetWorldVolume();

  const G4RegionStore* regStore = G4RegionStore::GetInstance();
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();
  unsigned int numLV = lvs->size();
  unsigned int nn = regStore->size();
  edm::LogVerbatim("SimG4CoreApplication")
      << "###RunManager: " << numPV << " PhysVolumes; " << numLV << " LogVolumes; " << nn << " Regions.";

  m_kernel->DefineWorldVolume(pworld, true);
  m_registry.dddWorldSignal_(world);

  if (m_pUseMagneticField) {
    // setup the magnetic field
    edm::ESHandle<MagneticField> pMF;
    es.get<IdealMagneticFieldRecord>().get(pMF);
    const GlobalPoint g(0., 0., 0.);

    sim::FieldBuilder fieldBuilder(pMF.product(), m_pField);
    CMSFieldManager* fieldManager = new CMSFieldManager();
    G4TransportationManager* tM = G4TransportationManager::GetTransportationManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build(fieldManager, tM->GetPropagatorInField());

    if (!m_FieldFile.empty()) {
      DumpMagneticField(tM->GetFieldManager()->GetDetectorField());
    }
  }

  // we need the track manager now
  m_trackManager = std::make_unique<SimTrackManager>();

  // attach sensitive detector
  AttachSD attach;
  auto sensDets = attach.create(es, catalog, m_p, m_trackManager.get(), m_registry);

  m_sensTkDets.swap(sensDets.first);
  m_sensCaloDets.swap(sensDets.second);

  edm::LogVerbatim("SimG4CoreApplication")
      << " RunManager: Sensitive Detector "
      << "building finished; found " << m_sensTkDets.size() << " Tk type Producers, and " << m_sensCaloDets.size()
      << " Calo type producers ";

  edm::ESHandle<HepPDT::ParticleDataTable> fTable;
  es.get<PDTRecord>().get(fTable);
  const HepPDT::ParticleDataTable* fPDGTable = &(*fTable);

  m_primaryTransformer = new PrimaryTransformer();

  std::unique_ptr<PhysicsListMakerBase> physicsMaker(
      PhysicsListFactory::get()->create(m_pPhysics.getParameter<std::string>("type")));
  if (physicsMaker.get() == nullptr) {
    throw edm::Exception(edm::errors::Configuration) << "Unable to find the Physics list requested";
  }
  m_physicsList = physicsMaker->make(m_pPhysics, m_registry);

  PhysicsList* phys = m_physicsList.get();
  if (phys == nullptr) {
    throw edm::Exception(edm::errors::Configuration) << "Physics list construction failed!";
  }

  // exotic particle physics
  double monopoleMass = m_pPhysics.getUntrackedParameter<double>("MonopoleMass", 0.);
  if (monopoleMass > 0.0) {
    phys->RegisterPhysics(new CMSMonopolePhysics(fPDGTable, m_pPhysics));
  }
  bool exotica = m_pPhysics.getUntrackedParameter<bool>("ExoticaTransport", false);
  if (exotica) {
    CMSExoticaPhysics exo(phys, m_pPhysics);
  }

  // adding GFlash, Russian Roulette for eletrons and gamma,
  // step limiters on top of any Physics Lists
  if (m_UseParametrisedEMPhysics)
    phys->RegisterPhysics(new ParametrisedEMPhysics("EMoptions", m_pPhysics));

  std::string tableDir = m_PhysicsTablesDir;
  if (m_RestorePhysicsTables) {
    m_physicsList->SetPhysicsTableRetrieved(tableDir);
  }
  edm::LogInfo("SimG4CoreApplication") << "RunManager: start initialisation of PhysicsList";

  m_physicsList->SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue") * CLHEP::cm);
  m_physicsList->SetCutsWithDefault();
  m_kernel->SetPhysics(phys);
  m_kernel->InitializePhysics();

  if (m_kernel->RunInitialization()) {
    m_managerInitialized = true;
  } else {
    throw edm::Exception(edm::errors::LogicError) << "G4RunManagerKernel initialization failed!";
  }

  if (m_StorePhysicsTables) {
    std::ostringstream dir;
    dir << tableDir << '\0';
    std::string cmd = std::string("/control/shell mkdir -p ") + tableDir;
    if (!std::ifstream(dir.str().c_str(), std::ios::in))
      G4UImanager::GetUIpointer()->ApplyCommand(cmd);
    m_physicsList->StorePhysicsTable(tableDir);
  }

  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_registry.beginOfJobSignal_(&aBeginOfJob);

  G4int sv = m_p.getUntrackedParameter<int>("SteppingVerbosity", 0);
  G4double elim = m_p.getUntrackedParameter<double>("StepVerboseThreshold", 0.1) * CLHEP::GeV;
  std::vector<int> ve = m_p.getUntrackedParameter<std::vector<int> >("VerboseEvents");
  std::vector<int> vn = m_p.getUntrackedParameter<std::vector<int> >("VertexNumber");
  std::vector<int> vt = m_p.getUntrackedParameter<std::vector<int> >("VerboseTracks");

  if (sv > 0) {
    m_sVerbose = std::make_unique<CMSSteppingVerbose>(sv, elim, ve, vn, vt);
  }
  initializeUserActions();

  if (!m_G4Commands.empty()) {
    G4cout << "RunManager: Requested UI commands: " << G4endl;
    for (unsigned it = 0; it < m_G4Commands.size(); ++it) {
      G4cout << "    " << m_G4Commands[it] << G4endl;
      G4UImanager::GetUIpointer()->ApplyCommand(m_G4Commands[it]);
    }
  }
  G4StateManager::GetStateManager()->SetNewState(G4State_Init);

  if (!m_WriteFile.empty()) {
    G4GDMLParser gdml;
    gdml.SetRegionExport(true);
    gdml.SetEnergyCutsExport(true);
    gdml.Write(m_WriteFile, pworld, true);
  }

  // G4Region dump file name
  auto regionFile = m_p.getUntrackedParameter<std::string>("FileNameRegions", "");

  // Geometry checks
  if (m_check || !regionFile.empty()) {
    CMSG4CheckOverlap check(m_g4overlap, regionFile, m_UIsession, pworld);
  }

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //  G4cout << "Output of G4ParticleTable DumpTable:" << G4endl;
  //  G4ParticleTable::GetParticleTable()->DumpTable("ALL");

  initializeRun();
  firstRun = false;
}

void RunManager::stopG4() {
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  if (!m_runTerminated) {
    terminateRun();
  }
}

void RunManager::produce(edm::Event& inpevt, const edm::EventSetup&) {
  m_currentEvent = generateEvent(inpevt);
  m_simEvent = new G4SimEvent;
  m_simEvent->hepEvent(m_generator->genEvent());
  m_simEvent->weight(m_generator->eventWeight());
  if (m_generator->genVertex() != nullptr) {
    m_simEvent->collisionPoint(math::XYZTLorentzVectorD(m_generator->genVertex()->x() / centimeter,
                                                        m_generator->genVertex()->y() / centimeter,
                                                        m_generator->genVertex()->z() / centimeter,
                                                        m_generator->genVertex()->t() / second));
  }
  if (m_currentEvent->GetNumberOfPrimaryVertex() == 0) {
    std::stringstream ss;
    ss << " RunManager::produce(): event " << inpevt.id().event() << " with no G4PrimaryVertices\n";
    throw SimG4Exception(ss.str());

    abortRun(false);
  } else {
    edm::LogInfo("SimG4CoreApplication") << "RunManager::produce: start Event " << inpevt.id().event() << " of weight "
                                         << m_simEvent->weight() << " with " << m_simEvent->nTracks() << " tracks and "
                                         << m_simEvent->nVertices() << " vertices, generated by "
                                         << m_simEvent->nGenParts() << " particles ";

    m_kernel->GetEventManager()->ProcessOneEvent(m_currentEvent);

    edm::LogInfo("SimG4CoreApplication") << " RunManager::produce: ended Event " << inpevt.id().event();
  }
}

G4Event* RunManager::generateEvent(edm::Event& inpevt) {
  if (m_currentEvent != nullptr) {
    delete m_currentEvent;
  }
  m_currentEvent = nullptr;
  if (m_simEvent != nullptr) {
    delete m_simEvent;
  }
  m_simEvent = nullptr;

  // 64 bits event ID in CMSSW converted into Geant4 event ID
  G4int evtid = (G4int)inpevt.id().event();
  G4Event* evt = new G4Event(evtid);

  edm::Handle<edm::HepMCProduct> HepMCEvt;

  inpevt.getByToken(m_HepMC, HepMCEvt);

  m_generator->setGenEvent(HepMCEvt->GetEvent());

  // required to reset the GenParticle Id for particles transported
  // along the beam pipe
  // to their original value for SimTrack creation
  resetGenParticleId(inpevt);

  if (!m_nonBeam) {
    m_generator->HepMC2G4(HepMCEvt->GetEvent(), evt);
  } else {
    m_generator->nonCentralEvent2G4(HepMCEvt->GetEvent(), evt);
  }

  return evt;
}

void RunManager::abortEvent() {
  if (m_runTerminated) {
    return;
  }
  G4Track* t = m_kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill);

  // CMS-specific act
  //
  TrackingAction* uta = (TrackingAction*)m_kernel->GetEventManager()->GetUserTrackingAction();
  uta->PostUserTrackingAction(t);

  m_currentEvent->SetEventAborted();
  m_kernel->GetEventManager()->GetStackManager()->clear();
  m_kernel->GetEventManager()->GetTrackingManager()->EventAborted();

  G4StateManager* stateManager = G4StateManager::GetStateManager();
  stateManager->SetNewState(G4State_GeomClosed);
}

void RunManager::initializeUserActions() {
  m_runInterface = new SimRunInterface(this, false);

  m_userRunAction = new RunAction(m_pRunAction, m_runInterface, true);
  Connect(m_userRunAction);

  G4EventManager* eventManager = m_kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  if (m_generator != nullptr) {
    EventAction* userEventAction =
        new EventAction(m_pEventAction, m_runInterface, m_trackManager.get(), m_sVerbose.get());
    Connect(userEventAction);
    eventManager->SetUserAction(userEventAction);

    TrackingAction* userTrackingAction = new TrackingAction(userEventAction, m_pTrackingAction, m_sVerbose.get());
    Connect(userTrackingAction);
    eventManager->SetUserAction(userTrackingAction);

    SteppingAction* userSteppingAction =
        new SteppingAction(userEventAction, m_pSteppingAction, m_sVerbose.get(), m_hasWatchers);
    Connect(userSteppingAction);
    eventManager->SetUserAction(userSteppingAction);

    eventManager->SetUserAction(new StackingAction(userTrackingAction, m_pStackingAction, m_sVerbose.get()));

  } else {
    edm::LogWarning("SimG4CoreApplication") << " RunManager: WARNING : "
                                            << "No generator; initialized "
                                            << "only RunAction!";
  }
}

void RunManager::initializeRun() {
  m_runInitialized = false;
  if (m_currentRun == nullptr) {
    m_currentRun = new G4Run();
  }
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (m_userRunAction != nullptr) {
    m_userRunAction->BeginOfRunAction(m_currentRun);
  }
  m_runAborted = false;
  m_runInitialized = true;
}

void RunManager::terminateRun() {
  if (m_runTerminated) {
    return;
  }
  if (m_userRunAction != nullptr) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    delete m_userRunAction;
    m_userRunAction = nullptr;
  }
  delete m_currentEvent;
  m_currentEvent = nullptr;
  delete m_simEvent;
  m_simEvent = nullptr;
  if (m_kernel != nullptr) {
    m_kernel->RunTermination();
  }
  m_runInitialized = false;
  m_runTerminated = true;
}

void RunManager::abortRun(bool softAbort) {
  if (m_runAborted) {
    return;
  }
  if (!softAbort) {
    abortEvent();
  }
  if (m_currentRun != nullptr) {
    delete m_currentRun;
    m_currentRun = nullptr;
  }
  terminateRun();
  m_runAborted = true;
}

void RunManager::resetGenParticleId(edm::Event& inpevt) {
  edm::Handle<edm::LHCTransportLinkContainer> theLHCTlink;
  inpevt.getByToken(m_LHCtr, theLHCTlink);
  if (theLHCTlink.isValid()) {
    m_trackManager->setLHCTransportLink(theLHCTlink.product());
  }
}

SimTrackManager* RunManager::GetSimTrackManager() { return m_trackManager.get(); }

void RunManager::Connect(RunAction* runAction) {
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void RunManager::Connect(EventAction* eventAction) {
  eventAction->m_beginOfEventSignal.connect(m_registry.beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_registry.endOfEventSignal_);
}

void RunManager::Connect(TrackingAction* trackingAction) {
  trackingAction->m_beginOfTrackSignal.connect(m_registry.beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_registry.endOfTrackSignal_);
}

void RunManager::Connect(SteppingAction* steppingAction) {
  steppingAction->m_g4StepSignal.connect(m_registry.g4StepSignal_);
}

void RunManager::DumpMagneticField(const G4Field* field) const {
  std::ofstream fout(m_FieldFile.c_str(), std::ios::out);
  if (fout.fail()) {
    edm::LogWarning("SimG4CoreApplication") << " RunManager WARNING : "
                                            << "error opening file <" << m_FieldFile << "> for magnetic field";
  } else {
    double rmax = 9000 * mm;
    double zmax = 16000 * mm;

    double dr = 5 * cm;
    double dz = 20 * cm;

    int nr = (int)(rmax / dr);
    int nz = 2 * (int)(zmax / dz);

    double r = 0.0;
    double z0 = -zmax;
    double z;

    double phi = 0.0;
    double cosf = cos(phi);
    double sinf = sin(phi);

    double point[4] = {0.0, 0.0, 0.0, 0.0};
    double bfield[3] = {0.0, 0.0, 0.0};

    fout << std::setprecision(6);
    for (int i = 0; i <= nr; ++i) {
      z = z0;
      for (int j = 0; j <= nz; ++j) {
        point[0] = r * cosf;
        point[1] = r * sinf;
        point[2] = z;
        field->GetFieldValue(point, bfield);
        fout << "R(mm)= " << r / mm << " phi(deg)= " << phi / degree << " Z(mm)= " << z / mm
             << "   Bz(tesla)= " << bfield[2] / tesla << " Br(tesla)= " << (bfield[0] * cosf + bfield[1] * sinf) / tesla
             << " Bphi(tesla)= " << (bfield[0] * sinf - bfield[1] * cosf) / tesla << G4endl;
        z += dz;
      }
      r += dr;
    }

    fout.close();
  }
}
