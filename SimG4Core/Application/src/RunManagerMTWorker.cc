#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/CustomUIsessionThreadPrefix.h"
#include "SimG4Core/Application/interface/CustomUIsessionToFile.h"
#include "SimG4Core/Application/interface/ExceptionHandler.h"

#include "SimG4Core/Geometry/interface/CustomUIsession.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "G4Event.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4WorkerThread.hh"
#include "G4WorkerRunManagerKernel.hh"
#include "G4StateManager.hh"
#include "G4TransportationManager.hh"
#include "G4Field.hh"
#include "G4FieldManager.hh"

#include <atomic>
#include <memory>

#include <thread>
#include <sstream>
#include <vector>

static std::once_flag applyOnce;
thread_local bool RunManagerMTWorker::dumpMF = false;

// from https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/3302/2.html
namespace {
  std::atomic<int> thread_counter{0};

  int get_new_thread_index() { return thread_counter++; }

  thread_local int s_thread_index = get_new_thread_index();

  int getThreadIndex() { return s_thread_index; }

  void createWatchers(const edm::ParameterSet& iP,
                      SimActivityRegistry* iReg,
                      std::vector<std::shared_ptr<SimWatcher> >& oWatchers,
                      std::vector<std::shared_ptr<SimProducer> >& oProds) {
    if (!iP.exists("Watchers")) {
      return;
    }

    std::vector<edm::ParameterSet> watchers = iP.getParameter<std::vector<edm::ParameterSet> >("Watchers");

    for (auto& watcher : watchers) {
      std::unique_ptr<SimWatcherMakerBase> maker(
          SimWatcherFactory::get()->create(watcher.getParameter<std::string>("type")));
      if (maker == nullptr) {
        throw edm::Exception(edm::errors::Configuration)
            << "Unable to find the requested Watcher <" << watcher.getParameter<std::string>("type");
      }
      std::shared_ptr<SimWatcher> watcherTemp;
      std::shared_ptr<SimProducer> producerTemp;
      maker->make(watcher, *(iReg), watcherTemp, producerTemp);
      oWatchers.push_back(watcherTemp);
      if (producerTemp) {
        oProds.push_back(producerTemp);
      }
    }
  }

  std::atomic<int> active_tlsdata{0};
  std::atomic<bool> tls_shutdown_timeout{false};
  std::atomic<int> n_tls_shutdown_task{0};
}  // namespace

struct RunManagerMTWorker::TLSData {
  std::unique_ptr<G4RunManagerKernel> kernel;  //must be deleted last
  std::unique_ptr<RunAction> userRunAction;
  std::unique_ptr<SimRunInterface> runInterface;
  std::unique_ptr<SimActivityRegistry> registry;
  std::unique_ptr<SimTrackManager> trackManager;
  std::vector<SensitiveTkDetector*> sensTkDets;
  std::vector<SensitiveCaloDetector*> sensCaloDets;
  std::vector<std::shared_ptr<SimWatcher> > watchers;
  std::vector<std::shared_ptr<SimProducer> > producers;
  //G4Run can only be deleted if there is a G4RunManager
  // on the thread where the G4Run is being deleted,
  // else it causes a segmentation fault
  G4Run* currentRun = nullptr;
  std::unique_ptr<G4Event> currentEvent;
  edm::RunNumber_t currentRunNumber = 0;
  bool threadInitialized = false;
  bool runTerminated = false;

  TLSData() { ++active_tlsdata; }

  ~TLSData() { --active_tlsdata; }
};

//This can not be a smart pointer since we must delete some of the members
// before leaving main() else we get a segmentation fault caused by accessing
// other 'singletons' after those singletons have been deleted. Instead we
// atempt to delete all TLS at RunManagerMTWorker destructor. If that fails for
// some reason, it is better to leak than cause a crash.
thread_local RunManagerMTWorker::TLSData* RunManagerMTWorker::m_tls{nullptr};

RunManagerMTWorker::RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : m_generator(iConfig.getParameter<edm::ParameterSet>("Generator")),
      m_InToken(iC.consumes<edm::HepMCProduct>(
          iConfig.getParameter<edm::ParameterSet>("Generator").getParameter<edm::InputTag>("HepMCProductLabel"))),
      m_theLHCTlinkToken(
          iC.consumes<edm::LHCTransportLinkContainer>(iConfig.getParameter<edm::InputTag>("theLHCTlinkTag"))),
      m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
      m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField")),
      m_LHCTransport(iConfig.getParameter<bool>("LHCTransport")),
      m_EvtMgrVerbosity(iConfig.getUntrackedParameter<int>("G4EventManagerVerbosity", 0)),
      m_pField(iConfig.getParameter<edm::ParameterSet>("MagneticField")),
      m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
      m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
      m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
      m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
      m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction")),
      m_pCustomUIsession(iConfig.getUntrackedParameter<edm::ParameterSet>("CustomUIsession")),
      m_p(iConfig),
      m_simEvent(nullptr),
      m_sVerbose(nullptr) {
  std::vector<edm::ParameterSet> watchers = iConfig.getParameter<std::vector<edm::ParameterSet> >("Watchers");
  m_hasWatchers = !watchers.empty();
  initializeTLS();
  int thisID = getThreadIndex();
  if (m_LHCTransport) {
    m_LHCToken = iC.consumes<edm::HepMCProduct>(edm::InputTag("LHCTransport"));
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker is constructed for the thread " << thisID;
}

RunManagerMTWorker::~RunManagerMTWorker() {
  ++n_tls_shutdown_task;
  resetTLS();

  {
    //make sure all tasks are done before continuing
    timespec s;
    s.tv_sec = 0;
    s.tv_nsec = 10000;
    while (n_tls_shutdown_task != 0) {
      nanosleep(&s, nullptr);
    }
  }
}

void RunManagerMTWorker::resetTLS() {
  m_tls = nullptr;

  if (active_tlsdata != 0 and not tls_shutdown_timeout) {
    ++n_tls_shutdown_task;
    //need to run tasks on each thread which has set the tls
    auto task = edm::make_functor_task(tbb::task::allocate_root(), []() { resetTLS(); });
    tbb::task::enqueue(*task);
    timespec s;
    s.tv_sec = 0;
    s.tv_nsec = 10000;
    //we do not want this thread to be used for a new task since it
    // has already cleared its structures. In order to fill all TBB
    // threads we wait for all TLSes to clear
    int count = 0;
    while (active_tlsdata.load() != 0 and ++count < 1000) {
      nanosleep(&s, nullptr);
    }
    if (count >= 1000) {
      tls_shutdown_timeout = true;
    }
  }
  --n_tls_shutdown_task;
}

void RunManagerMTWorker::endRun() {
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::endRun for the thread " << thisID;
  terminateRun();
}

void RunManagerMTWorker::initializeTLS() {
  if (m_tls) {
    return;
  }

  m_tls = new TLSData();
  m_tls->registry = std::make_unique<SimActivityRegistry>();

  edm::Service<SimActivityRegistry> otherRegistry;
  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  int thisID = getThreadIndex();
  if (otherRegistry) {
    m_tls->registry->connect(*otherRegistry);
    if (thisID > 0) {
      throw edm::Exception(edm::errors::Configuration)
          << "SimActivityRegistry service (i.e. visualization) is not supported for more than 1 thread. "
          << " \n If this use case is needed, RunManagerMTWorker has to be updated.";
    }
  }
  if (m_hasWatchers) {
    createWatchers(m_p, m_tls->registry.get(), m_tls->watchers, m_tls->producers);
  }
}

void RunManagerMTWorker::initializeG4(RunManagerMT* runManagerMaster, const edm::EventSetup& es) {
  // I guess everything initialized here should be in thread_local storage
  initializeTLS();
  if (m_tls->threadInitialized)
    return;

  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::initializeThread " << thisID << " is started";

  // Initialize per-thread output
  G4Threading::G4SetThreadId(thisID);
  G4UImanager::GetUIpointer()->SetUpForAThread(thisID);
  const std::string& uitype = m_pCustomUIsession.getUntrackedParameter<std::string>("Type", "MessageLogger");
  if (uitype == "MessageLogger") {
    new CustomUIsession();
  } else if (uitype == "MessageLoggerThreadPrefix") {
    new CustomUIsessionThreadPrefix(m_pCustomUIsession.getUntrackedParameter<std::string>("ThreadPrefix", ""), thisID);
  } else if (uitype == "FilePerThread") {
    new CustomUIsessionToFile(m_pCustomUIsession.getUntrackedParameter<std::string>("ThreadFile", ""), thisID);
  } else {
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid value of CustomUIsession.Type '" << uitype
        << "', valid are MessageLogger, MessageLoggerThreadPrefix, FilePerThread";
  }

  // Initialize worker part of shared resources (geometry, physics)
  G4WorkerThread::BuildGeometryAndPhysicsVector();

  // Create worker run manager
  m_tls->kernel.reset(G4WorkerRunManagerKernel::GetRunManagerKernel());
  if (!m_tls->kernel) {
    m_tls->kernel = std::make_unique<G4WorkerRunManagerKernel>();
  }

  // Define G4 exception handler
  G4StateManager::GetStateManager()->SetExceptionHandler(new ExceptionHandler());

  // Set the geometry for the worker, share from master
  auto worldPV = runManagerMaster->world().GetWorldVolume();
  m_tls->kernel->WorkerDefineWorldVolume(worldPV);
  G4TransportationManager* tM = G4TransportationManager::GetTransportationManager();
  tM->SetWorldForTracking(worldPV);

  // we need the track manager now
  m_tls->trackManager = std::make_unique<SimTrackManager>();

  // setup the magnetic field
  if (m_pUseMagneticField) {
    const GlobalPoint g(0., 0., 0.);

    edm::ESHandle<MagneticField> pMF;
    es.get<IdealMagneticFieldRecord>().get(pMF);

    sim::FieldBuilder fieldBuilder(pMF.product(), m_pField);
    CMSFieldManager* fieldManager = new CMSFieldManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build(fieldManager, tM->GetPropagatorInField());

    std::string fieldFile = m_p.getUntrackedParameter<std::string>("FileNameField", "");
    if (!fieldFile.empty()) {
      std::call_once(applyOnce, []() { dumpMF = true; });
      if (dumpMF) {
        edm::LogVerbatim("SimG4CoreApplication") << " RunManagerMTWorker: Dump magnetic field to file " << fieldFile;
        DumpMagneticField(tM->GetFieldManager()->GetDetectorField(), fieldFile);
      }
    }
  }

  // attach sensitive detector
  AttachSD attach;
  auto sensDets =
      attach.create(es, runManagerMaster->catalog(), m_p, m_tls->trackManager.get(), *(m_tls->registry.get()));

  m_tls->sensTkDets.swap(sensDets.first);
  m_tls->sensCaloDets.swap(sensDets.second);

  edm::LogVerbatim("SimG4CoreApplication")
      << " RunManagerMTWorker: Sensitive Detector building finished; found " << m_tls->sensTkDets.size()
      << " Tk type Producers, and " << m_tls->sensCaloDets.size() << " Calo type producers ";

  // Set the physics list for the worker, share from master
  PhysicsList* physicsList = runManagerMaster->physicsListForWorker();

  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker: start initialisation of PhysicsList for the thread";

  // Geant4 UI commands in PreInit state
  if (!runManagerMaster->G4Commands().empty()) {
    G4cout << "RunManagerMTWorker: Requested UI commands: " << G4endl;
    for (const std::string& command : runManagerMaster->G4Commands()) {
      G4cout << "          " << command << G4endl;
      G4UImanager::GetUIpointer()->ApplyCommand(command);
    }
  }
  G4StateManager::GetStateManager()->SetNewState(G4State_Init);

  physicsList->InitializeWorker();
  m_tls->kernel->SetPhysics(physicsList);
  m_tls->kernel->InitializePhysics();

  const bool kernelInit = m_tls->kernel->RunInitialization();
  if (!kernelInit) {
    throw edm::Exception(edm::errors::Configuration) << "RunManagerMTWorker: Geant4 kernel initialization failed";
  }
  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_tls->registry->beginOfJobSignal_(&aBeginOfJob);

  G4int sv = m_p.getUntrackedParameter<int>("SteppingVerbosity", 0);
  G4double elim = m_p.getUntrackedParameter<double>("StepVerboseThreshold", 0.1) * CLHEP::GeV;
  std::vector<int> ve = m_p.getUntrackedParameter<std::vector<int> >("VerboseEvents");
  std::vector<int> vn = m_p.getUntrackedParameter<std::vector<int> >("VertexNumber");
  std::vector<int> vt = m_p.getUntrackedParameter<std::vector<int> >("VerboseTracks");

  if (sv > 0) {
    m_sVerbose = std::make_unique<CMSSteppingVerbose>(sv, elim, ve, vn, vt);
  }
  initializeUserActions();

  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::initializeThread done for the thread " << thisID;

  G4StateManager::GetStateManager()->SetNewState(G4State_Idle);
  m_tls->threadInitialized = true;
}

void RunManagerMTWorker::initializeUserActions() {
  m_tls->runInterface = std::make_unique<SimRunInterface>(this, false);
  m_tls->userRunAction = std::make_unique<RunAction>(m_pRunAction, m_tls->runInterface.get(), false);
  m_tls->userRunAction->SetMaster(false);
  Connect(m_tls->userRunAction.get());

  G4EventManager* eventManager = m_tls->kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  EventAction* userEventAction =
      new EventAction(m_pEventAction, m_tls->runInterface.get(), m_tls->trackManager.get(), m_sVerbose.get());
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
}

void RunManagerMTWorker::Connect(RunAction* runAction) {
  runAction->m_beginOfRunSignal.connect(m_tls->registry->beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_tls->registry->endOfRunSignal_);
}

void RunManagerMTWorker::Connect(EventAction* eventAction) {
  eventAction->m_beginOfEventSignal.connect(m_tls->registry->beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_tls->registry->endOfEventSignal_);
}

void RunManagerMTWorker::Connect(TrackingAction* trackingAction) {
  trackingAction->m_beginOfTrackSignal.connect(m_tls->registry->beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_tls->registry->endOfTrackSignal_);
}

void RunManagerMTWorker::Connect(SteppingAction* steppingAction) {
  steppingAction->m_g4StepSignal.connect(m_tls->registry->g4StepSignal_);
}

SimTrackManager* RunManagerMTWorker::GetSimTrackManager() {
  initializeTLS();
  return m_tls->trackManager.get();
}
std::vector<SensitiveTkDetector*>& RunManagerMTWorker::sensTkDetectors() {
  initializeTLS();
  return m_tls->sensTkDets;
}
std::vector<SensitiveCaloDetector*>& RunManagerMTWorker::sensCaloDetectors() {
  initializeTLS();
  return m_tls->sensCaloDets;
}
std::vector<std::shared_ptr<SimProducer> >& RunManagerMTWorker::producers() {
  initializeTLS();
  return m_tls->producers;
}

void RunManagerMTWorker::initializeRun() {
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::initializeRun " << thisID << " is started";
  m_tls->currentRun = new G4Run();
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (m_tls->userRunAction) {
    m_tls->userRunAction->BeginOfRunAction(m_tls->currentRun);
  }
}

void RunManagerMTWorker::terminateRun() {
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::terminateRun ";
  if (!m_tls || m_tls->runTerminated) {
    return;
  }
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::terminateRun " << thisID << " is started";
  if (m_tls->userRunAction) {
    m_tls->userRunAction->EndOfRunAction(m_tls->currentRun);
    m_tls->userRunAction.reset();
  }
  m_tls->currentEvent.reset();
  m_simEvent = nullptr;

  if (m_tls->kernel) {
    m_tls->kernel->RunTermination();
  }

  m_tls->runTerminated = true;
}

std::unique_ptr<G4SimEvent> RunManagerMTWorker::produce(const edm::Event& inpevt,
                                                        const edm::EventSetup& es,
                                                        RunManagerMT& runManagerMaster) {
  // The initialization and begin/end run is a bit convoluted due to
  // - Geant4 deals per-thread
  // - OscarMTProducer deals per-stream
  // and framework/TBB is free to schedule work in streams to the
  // threads as it likes.
  //
  // We have to do the per-thread initialization, and per-thread
  // per-run initialization here by ourselves.

  if (!(m_tls && m_tls->threadInitialized)) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "RunManagerMTWorker::produce(): stream " << inpevt.streamID() << " thread " << getThreadIndex()
        << " Geant4 initialisation for this thread";
    initializeG4(&runManagerMaster, es);
    m_tls->threadInitialized = true;
  }
  // Initialize run
  if (inpevt.id().run() != m_tls->currentRunNumber) {
    if (m_tls->currentRunNumber != 0 && !m_tls->runTerminated) {
      // If previous run in this thread was not terminated via endRun() call, terminate it now
      terminateRun();
    }
    initializeRun();
    m_tls->currentRunNumber = inpevt.id().run();
  }
  m_tls->runInterface->setRunManagerMTWorker(this);  // For UserActions

  m_tls->currentEvent.reset(generateEvent(inpevt));

  auto simEvent = std::make_unique<G4SimEvent>();
  m_simEvent = simEvent.get();
  m_simEvent->hepEvent(m_generator.genEvent());
  m_simEvent->weight(m_generator.eventWeight());
  if (m_generator.genVertex() != nullptr) {
    auto genVertex = m_generator.genVertex();
    m_simEvent->collisionPoint(math::XYZTLorentzVectorD(genVertex->x() / CLHEP::cm,
                                                        genVertex->y() / CLHEP::cm,
                                                        genVertex->z() / CLHEP::cm,
                                                        genVertex->t() / CLHEP::second));
  }
  if (m_tls->currentEvent->GetNumberOfPrimaryVertex() == 0) {
    std::stringstream ss;
    ss << "RunManagerMTWorker::produce: event " << inpevt.id().event() << " with no G4PrimaryVertices \n";
    throw SimG4Exception(ss.str());

  } else {
    if (!m_tls->kernel) {
      std::stringstream ss;
      ss << "RunManagerMTWorker::produce: "
         << " no G4WorkerRunManagerKernel yet for thread index" << getThreadIndex() << ", id " << std::hex
         << std::this_thread::get_id() << " \n";
      throw SimG4Exception(ss.str());
    }

    edm::LogVerbatim("SimG4CoreApplication")
        << "RunManagerMTWorker::produce: start EventID=" << inpevt.id().event() << " StreamID=" << inpevt.streamID()
        << " threadIndex=" << getThreadIndex() << " weight=" << m_simEvent->weight() << "; "
        << m_tls->currentEvent->GetNumberOfPrimaryVertex() << " vertices for Geant4; generator produced "
        << m_simEvent->nGenParts() << " particles.";

    m_tls->kernel->GetEventManager()->ProcessOneEvent(m_tls->currentEvent.get());
  }

  //remove memory only needed during event processing
  m_tls->currentEvent.reset();

  for (auto& sd : m_tls->sensCaloDets) {
    sd->reset();
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::produce: ended Event " << inpevt.id().event();

  m_simEvent = nullptr;
  return simEvent;
}

void RunManagerMTWorker::abortEvent() {
  if (m_tls->runTerminated) {
    return;
  }
  G4Track* t = m_tls->kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill);

  // CMS-specific act
  //
  TrackingAction* uta = static_cast<TrackingAction*>(m_tls->kernel->GetEventManager()->GetUserTrackingAction());
  uta->PostUserTrackingAction(t);

  m_tls->currentEvent->SetEventAborted();
  m_tls->kernel->GetEventManager()->GetStackManager()->clear();
  m_tls->kernel->GetEventManager()->GetTrackingManager()->EventAborted();
}

void RunManagerMTWorker::abortRun(bool softAbort) {
  if (!softAbort) {
    abortEvent();
  }
  m_tls->currentRun = nullptr;
  terminateRun();
}

G4Event* RunManagerMTWorker::generateEvent(const edm::Event& inpevt) {
  m_tls->currentEvent.reset();
  m_simEvent = nullptr;

  // 64 bits event ID in CMSSW converted into Geant4 event ID
  G4int evtid = (G4int)inpevt.id().event();
  G4Event* evt = new G4Event(evtid);

  edm::Handle<edm::HepMCProduct> HepMCEvt;
  inpevt.getByToken(m_InToken, HepMCEvt);

  m_generator.setGenEvent(HepMCEvt->GetEvent());

  // required to reset the GenParticle Id for particles transported
  // along the beam pipe
  // to their original value for SimTrack creation
  resetGenParticleId(inpevt);

  if (!m_nonBeam) {
    m_generator.HepMC2G4(HepMCEvt->GetEvent(), evt);
    if (m_LHCTransport) {
      edm::Handle<edm::HepMCProduct> LHCMCEvt;
      inpevt.getByToken(m_LHCToken, LHCMCEvt);
      m_generator.nonCentralEvent2G4(LHCMCEvt->GetEvent(), evt);
    }
  } else {
    m_generator.nonCentralEvent2G4(HepMCEvt->GetEvent(), evt);
  }

  return evt;
}

void RunManagerMTWorker::resetGenParticleId(const edm::Event& inpevt) {
  edm::Handle<edm::LHCTransportLinkContainer> theLHCTlink;
  inpevt.getByToken(m_theLHCTlinkToken, theLHCTlink);
  if (theLHCTlink.isValid()) {
    m_tls->trackManager->setLHCTransportLink(theLHCTlink.product());
  }
}

void RunManagerMTWorker::DumpMagneticField(const G4Field* field, const std::string& file) const {
  std::ofstream fout(file.c_str(), std::ios::out);
  if (fout.fail()) {
    edm::LogWarning("SimG4CoreApplication")
        << " RunManager WARNING : error opening file <" << file << "> for magnetic field";
  } else {
    // CMS magnetic field volume
    double rmax = 9000 * mm;
    double zmax = 24000 * mm;

    double dr = 1 * cm;
    double dz = 5 * cm;

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
