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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/SensitiveDetector/interface/sensitiveDetectorMakers.h"

#include "G4Timer.hh"
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
#include "tbb/task_arena.h"

static std::once_flag applyOnce;

// from https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/3302/2.html
namespace {
  std::atomic<int> thread_counter{0};

  int get_new_thread_index() { return thread_counter++; }

  bool createWatchers(const edm::ParameterSet& iP,
                      SimActivityRegistry* iReg,
                      std::vector<SimWatcher*>& oWatchers,
                      std::vector<SimProducer*>& oProds,
                      int threadID) {
    std::vector<edm::ParameterSet> watchers = iP.getParameter<std::vector<edm::ParameterSet>>("Watchers");

    // Watchers following old interface applicable only to 1-thread run
    if (watchers.empty()) {
      return false;
    }

    for (auto& watcher : watchers) {
      std::unique_ptr<SimWatcherMakerBase> maker(
          SimWatcherFactory::get()->create(watcher.getParameter<std::string>("type")));
      if (maker == nullptr) {
        throw edm::Exception(edm::errors::Configuration)
            << "RunManagerMTWorker::createWatchers: "
            << "Unable to find the requested Watcher " << watcher.getParameter<std::string>("type");
      } else {
        SimWatcher* newWatcher = maker->makeWatcher(watcher, *(iReg));
        if (nullptr != newWatcher) {
          if (!newWatcher->isMT() && 0 < threadID) {
            throw edm::Exception(edm::errors::Configuration)
                << "RunManagerMTWorker::createWatchers: "
                << "Unable to use Watcher " << watcher.getParameter<std::string>("type") << " if number of threads > 1";

          } else {
            oWatchers.push_back(newWatcher);
            SimProducer* producerTemp = static_cast<SimProducer*>(newWatcher);
            if (nullptr != producerTemp) {
              oProds.push_back(producerTemp);
            }
          }
        }
      }
    }
    return (!oWatchers.empty());
  }
};  // namespace

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
      m_thread_index{get_new_thread_index()},
      m_pField(iConfig.getParameter<edm::ParameterSet>("MagneticField")),
      m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
      m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
      m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
      m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
      m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction")),
      m_pCustomUIsession(iConfig.getUntrackedParameter<edm::ParameterSet>("CustomUIsession")),
      m_p(iConfig) {
  std::vector<std::string> onlySDs = iConfig.getParameter<std::vector<std::string>>("OnlySDs");
  m_sdMakers = sim::sensitiveDetectorMakers(m_p, iC, onlySDs);
  m_registry = std::make_unique<SimActivityRegistry>();

  int thisID = getThreadIndex();

  // Look for an outside SimActivityRegistry this is used by the visualization code
  edm::Service<SimActivityRegistry> otherRegistry;
  if (otherRegistry && 0 == thisID) {
    m_registry->connect(*otherRegistry);
  }

  m_hasWatchers = createWatchers(m_p, m_registry.get(), m_watchers, m_producers, thisID);
  if (m_hasWatchers) {
    for (auto& watcher : m_watchers) {
      watcher->registerConsumes(iC);
    }
  }

  if (m_LHCTransport) {
    m_LHCToken = iC.consumes<edm::HepMCProduct>(edm::InputTag("LHCTransport"));
  }
  if (m_pUseMagneticField) {
    m_MagField = iC.esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>();
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker is constructed for the thread " << thisID;
  unsigned int k = 0;
  for (std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>>::const_iterator itr =
           m_sdMakers.begin();
       itr != m_sdMakers.end();
       ++itr, ++k)
    edm::LogVerbatim("SimG4CoreApplication") << "SD[" << k << "] " << itr->first;
}

RunManagerMTWorker::~RunManagerMTWorker() {
  for (auto& watcher : m_watchers) {
    delete watcher;
  }
}

void RunManagerMTWorker::beginRun(edm::EventSetup const& es) {
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::beginRun for the thread " << thisID;
  for (auto& maker : m_sdMakers) {
    maker.second->beginRun(es);
  }
  if (m_pUseMagneticField) {
    m_pMagField = &es.getData(m_MagField);
  }
  if (m_hasWatchers) {
    for (auto& watcher : m_watchers) {
      watcher->beginRun(es);
    }
  }
}

void RunManagerMTWorker::endRun() {
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::endRun for the thread " << thisID;
  terminateRun();
}

void RunManagerMTWorker::initializeG4(RunManagerMT* runManagerMaster, const edm::EventSetup& es) {
  if (m_threadInitialized)
    return;

  G4Timer timer;
  timer.Start();

  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::initializeG4 " << thisID << " is started";

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
        << "RunManagerMTWorker::initializeG4: Invalid value of CustomUIsession.Type '" << uitype
        << "', valid are MessageLogger, MessageLoggerThreadPrefix, FilePerThread";
  }

  // Initialize worker part of shared resources (geometry, physics)
  G4WorkerThread::BuildGeometryAndPhysicsVector();

  // Create worker run manager
  m_kernel.reset(G4WorkerRunManagerKernel::GetRunManagerKernel());
  if (nullptr == m_kernel) {
    m_kernel = std::make_unique<G4WorkerRunManagerKernel>();
  }

  // Define G4 exception handler
  double th = m_p.getParameter<double>("ThresholdForGeometryExceptions") * CLHEP::GeV;
  G4StateManager::GetStateManager()->SetExceptionHandler(new ExceptionHandler(th));

  // Set the geometry for the worker, share from master
  auto worldPV = runManagerMaster->world().GetWorldVolume();
  m_kernel->WorkerDefineWorldVolume(worldPV);
  G4TransportationManager* tM = G4TransportationManager::GetTransportationManager();
  tM->SetWorldForTracking(worldPV);

  // we need the track manager now
  m_trackManager = std::make_unique<SimTrackManager>();

  // setup the magnetic field
  if (m_pUseMagneticField) {
    const GlobalPoint g(0.f, 0.f, 0.f);

    sim::FieldBuilder fieldBuilder(m_pMagField, m_pField);

    CMSFieldManager* fieldManager = new CMSFieldManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build(fieldManager, tM->GetPropagatorInField());

    std::string fieldFile = m_p.getUntrackedParameter<std::string>("FileNameField", "");
    if (!fieldFile.empty()) {
      std::call_once(applyOnce, [this]() { m_dumpMF = true; });
      if (m_dumpMF) {
        edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker: Dump magnetic field to file " << fieldFile;
        DumpMagneticField(tM->GetFieldManager()->GetDetectorField(), fieldFile);
      }
    }
  }

  // attach sensitive detector
  auto sensDets =
      sim::attachSD(m_sdMakers, es, runManagerMaster->catalog(), m_p, m_trackManager.get(), *(m_registry.get()));

  m_sensTkDets.swap(sensDets.first);
  m_sensCaloDets.swap(sensDets.second);

  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMTWorker: Sensitive Detectors are built in thread " << thisID << " found " << m_sensTkDets.size()
      << " Tk type SD, and " << m_sensCaloDets.size() << " Calo type SD";

  // Set the physics list for the worker, share from master
  PhysicsList* physicsList = runManagerMaster->physicsListForWorker();

  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMTWorker: start initialisation of PhysicsList for the thread " << thisID;

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
  m_kernel->SetPhysics(physicsList);
  m_kernel->InitializePhysics();

  if (!m_kernel->RunInitialization()) {
    throw edm::Exception(edm::errors::Configuration)
        << "RunManagerMTWorker::initializeG4: Geant4 kernel initialization failed in thread " << thisID;
  }
  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_registry->beginOfJobSignal_(&aBeginOfJob);

  G4int sv = m_p.getUntrackedParameter<int>("SteppingVerbosity", 0);
  G4double elim = m_p.getUntrackedParameter<double>("StepVerboseThreshold", 0.1) * CLHEP::GeV;
  std::vector<int> ve = m_p.getUntrackedParameter<std::vector<int>>("VerboseEvents");
  std::vector<int> vn = m_p.getUntrackedParameter<std::vector<int>>("VertexNumber");
  std::vector<int> vt = m_p.getUntrackedParameter<std::vector<int>>("VerboseTracks");

  if (sv > 0) {
    m_sVerbose = std::make_unique<CMSSteppingVerbose>(sv, elim, ve, vn, vt);
  }
  initializeUserActions();

  G4StateManager::GetStateManager()->SetNewState(G4State_Idle);
  m_threadInitialized = true;

  timer.Stop();
  edm::LogVerbatim("SimG4CoreApplication")
      << "RunManagerMTWorker::initializeThread done for the thread " << thisID << "  " << timer;
}

void RunManagerMTWorker::initializeUserActions() {
  m_runInterface = std::make_unique<SimRunInterface>(this, false);
  m_userRunAction = std::make_unique<RunAction>(m_pRunAction, m_runInterface.get(), false);
  m_userRunAction->SetMaster(false);
  Connect(m_userRunAction.get());

  G4EventManager* eventManager = m_kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  EventAction* userEventAction =
      new EventAction(m_pEventAction, m_runInterface.get(), m_trackManager.get(), m_sVerbose.get());
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
  runAction->m_beginOfRunSignal.connect(m_registry->beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry->endOfRunSignal_);
}

void RunManagerMTWorker::Connect(EventAction* eventAction) {
  eventAction->m_beginOfEventSignal.connect(m_registry->beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_registry->endOfEventSignal_);
}

void RunManagerMTWorker::Connect(TrackingAction* trackingAction) {
  trackingAction->m_beginOfTrackSignal.connect(m_registry->beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_registry->endOfTrackSignal_);
}

void RunManagerMTWorker::Connect(SteppingAction* steppingAction) {
  steppingAction->m_g4StepSignal.connect(m_registry->g4StepSignal_);
}

void RunManagerMTWorker::initializeRun() {
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::initializeRun " << thisID << " is started";
  m_currentRun = new G4Run();
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (nullptr != m_userRunAction) {
    m_userRunAction->BeginOfRunAction(m_currentRun);
  }
}

void RunManagerMTWorker::terminateRun() {
  if (m_runTerminated) {
    return;
  }
  int thisID = getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::terminateRun " << thisID << " is started";
  if (nullptr != m_userRunAction) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    m_userRunAction.reset();
  }
  m_currentEvent.reset();
  m_simEvent = nullptr;

  if (nullptr != m_kernel) {
    m_kernel->RunTermination();
  }
  m_runTerminated = true;
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

  assert(m_threadInitialized);
  // Initialize run
  if (inpevt.id().run() != m_currentRunNumber) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "RunID= " << inpevt.id().run() << "  and local RunID= " << m_currentRunNumber;
    if (m_currentRunNumber != 0 && !m_runTerminated) {
      // If previous run in this thread was not terminated via endRun() call,
      // terminate it now
      terminateRun();
    }
    initializeRun();
    m_currentRunNumber = inpevt.id().run();
  }
  m_runInterface->setRunManagerMTWorker(this);  // For UserActions

  m_currentEvent.reset(generateEvent(inpevt));

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
  if (m_currentEvent->GetNumberOfPrimaryVertex() == 0) {
    std::stringstream ss;
    ss << "RunManagerMTWorker::produce: event " << inpevt.id().event() << " with no G4PrimaryVertices \n";
    throw SimG4Exception(ss.str());

  } else {
    edm::LogVerbatim("SimG4CoreApplication")
        << "RunManagerMTWorker::produce: start EventID=" << inpevt.id().event() << " StreamID=" << inpevt.streamID()
        << " threadIndex=" << getThreadIndex() << " weight=" << m_simEvent->weight() << "; "
        << m_currentEvent->GetNumberOfPrimaryVertex() << " vertices for Geant4; generator produced "
        << m_simEvent->nGenParts() << " particles.";

    m_kernel->GetEventManager()->ProcessOneEvent(m_currentEvent.get());
  }

  //remove memory only needed during event processing
  m_currentEvent.reset();

  for (auto& sd : m_sensCaloDets) {
    sd->reset();
  }
  edm::LogVerbatim("SimG4CoreApplication") << "RunManagerMTWorker::produce: ended Event " << inpevt.id().event();

  m_simEvent = nullptr;
  return simEvent;
}

void RunManagerMTWorker::abortEvent() {
  if (m_runTerminated) {
    return;
  }
  G4Track* t = m_kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill);

  // CMS-specific act
  //
  TrackingAction* uta = static_cast<TrackingAction*>(m_kernel->GetEventManager()->GetUserTrackingAction());
  uta->PostUserTrackingAction(t);

  m_currentEvent->SetEventAborted();
  m_kernel->GetEventManager()->GetStackManager()->clear();
  m_kernel->GetEventManager()->GetTrackingManager()->EventAborted();
}

void RunManagerMTWorker::abortRun(bool softAbort) {
  if (!softAbort)
    abortEvent();

  m_currentRun = nullptr;
  terminateRun();
}

G4Event* RunManagerMTWorker::generateEvent(const edm::Event& inpevt) {
  m_currentEvent.reset();
  m_simEvent = nullptr;

  // 64 bits event ID in CMSSW converted into Geant4 event ID
  G4int evtid = inpevt.id().event();
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
    m_trackManager->setLHCTransportLink(theLHCTlink.product());
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
