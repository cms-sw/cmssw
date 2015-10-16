#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"
#include "SimG4Core/Application/interface/CustomUIsessionThreadPrefix.h"
#include "SimG4Core/Application/interface/CustomUIsessionToFile.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "G4Event.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4WorkerThread.hh"
#include "G4WorkerRunManagerKernel.hh"
#include "G4StateManager.hh"
#include "G4TransportationManager.hh"

#include <atomic>
#include <thread>
#include <sstream>


// from https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/3302/2.html
namespace {
  static std::atomic<int> thread_counter{ 0 };

  int get_new_thread_index() { 
    return thread_counter++;
  }

  static thread_local int s_thread_index = get_new_thread_index();

  int getThreadIndex() { return s_thread_index; }

  void createWatchers(const edm::ParameterSet& iP,
                      SimActivityRegistry& iReg,
                      std::vector<std::shared_ptr<SimWatcher> >& oWatchers,
                      std::vector<std::shared_ptr<SimProducer> >& oProds,
                      int thisThreadID
                      )
  {
    using namespace std;
    using namespace edm;
    if(!iP.exists("Watchers"))
      return;

    vector<ParameterSet> watchers = iP.getParameter<vector<ParameterSet> >("Watchers");

    if(!watchers.empty() && thisThreadID > 0) {
      throw cms::Exception("Unsupported") << "SimWatchers are not supported for more than 1 thread. If this use case is needed, RunManagerMTWorker has to be updated, and SimWatchers and SimProducers have to be made thread safe.";
    }

    for(vector<ParameterSet>::iterator itWatcher = watchers.begin();
        itWatcher != watchers.end();
        ++itWatcher) {
      std::unique_ptr<SimWatcherMakerBase> maker(
        SimWatcherFactory::get()->create(itWatcher->getParameter<std::string>("type"))
      );
      if(maker.get()==0) {
        throw SimG4Exception("Unable to find the requested Watcher");
      }

      std::shared_ptr<SimWatcher> watcherTemp;
      std::shared_ptr<SimProducer> producerTemp;
      maker->make(*itWatcher,iReg,watcherTemp,producerTemp);
      oWatchers.push_back(watcherTemp);
      if(producerTemp) {
        oProds.push_back(producerTemp);
      }
    }
  }
}

struct RunManagerMTWorker::TLSData {
  std::unique_ptr<CustomUIsession> UIsession;
  std::unique_ptr<RunAction> userRunAction;
  std::unique_ptr<SimRunInterface> runInterface;
  SimActivityRegistry registry;
  std::unique_ptr<SimTrackManager> trackManager;
  std::vector<SensitiveTkDetector*> sensTkDets;
  std::vector<SensitiveCaloDetector*> sensCaloDets;
  std::vector<std::shared_ptr<SimWatcher> > watchers;
  std::vector<std::shared_ptr<SimProducer> > producers;
  std::unique_ptr<sim::FieldBuilder> fieldBuilder;
  std::unique_ptr<G4Run> currentRun;
  std::unique_ptr<G4Event> currentEvent;
  edm::RunNumber_t currentRunNumber = 0;
  bool threadInitialized = false;
  bool runTerminated = false;
};

thread_local RunManagerMTWorker::TLSData *RunManagerMTWorker::m_tls = nullptr;

RunManagerMTWorker::RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  m_generator(iConfig.getParameter<edm::ParameterSet>("Generator")),
  m_InToken(iC.consumes<edm::HepMCProduct>(iConfig.getParameter<edm::ParameterSet>("Generator").getParameter<std::string>("HepMCProductLabel"))),
  m_theLHCTlinkToken(iC.consumes<edm::LHCTransportLinkContainer>(iConfig.getParameter<edm::InputTag>("theLHCTlinkTag"))),
  m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
  m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField")),
  m_EvtMgrVerbosity(iConfig.getUntrackedParameter<int>("G4EventManagerVerbosity",0)),
  m_pField(iConfig.getParameter<edm::ParameterSet>("MagneticField")),
  m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
  m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
  m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
  m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
  m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction")),
  m_pCustomUIsession(iConfig.getUntrackedParameter<edm::ParameterSet>("CustomUIsession")),
  m_p(iConfig)
{
  initializeTLS();
}

RunManagerMTWorker::~RunManagerMTWorker() {
  if(!(m_tls && m_tls->runTerminated)) { terminateRun(); }
  // RunManagerMT has 'delete m_runInterface' in the destructor, but
  // doesn't make much sense here because it is thread_local and we're
  // not guaranteed to run the destructor on each of the threads.
}

void RunManagerMTWorker::endRun() {
  terminateRun();
}

void RunManagerMTWorker::initializeTLS() {
  if(m_tls)
    return;
  m_tls = new TLSData;

  edm::Service<SimActivityRegistry> otherRegistry;
  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  int thisID = getThreadIndex();
  if(otherRegistry){
    m_tls->registry.connect(*otherRegistry);
    if(thisID > 0) {
      throw cms::Exception("Unsupported") << "SimActivityRegistry service (i.e. visualization) is not supported for more than 1 thread. If this use case is needed, RunManagerMTWorker has to be updated.";
    }
  }

  createWatchers(m_p, m_tls->registry, m_tls->watchers, m_tls->producers, thisID);
}

void RunManagerMTWorker::initializeThread(const RunManagerMT& runManagerMaster, const edm::EventSetup& es) {
  // I guess everything initialized here should be in thread_local storage
  initializeTLS();

  int thisID = getThreadIndex();

  // Initialize per-thread output
  G4Threading::G4SetThreadId( thisID );
  G4UImanager::GetUIpointer()->SetUpForAThread( thisID );
  const std::string& uitype = m_pCustomUIsession.getUntrackedParameter<std::string>("Type");
  if(uitype == "MessageLogger") {
    m_tls->UIsession.reset(new CustomUIsession());
  }
  else if(uitype == "MessageLoggerThreadPrefix") {
    m_tls->UIsession.reset(new CustomUIsessionThreadPrefix(m_pCustomUIsession.getUntrackedParameter<std::string>("ThreadPrefix"), thisID));
  }
  else if(uitype == "FilePerThread") {
    m_tls->UIsession.reset(new CustomUIsessionToFile(m_pCustomUIsession.getUntrackedParameter<std::string>("ThreadFile"), thisID));
  }
  else {
    throw cms::Exception("Configuration") << "Invalid value of CustomUIsession.Type '" << uitype << "', valid are MessageLogger, MessageLoggerThreadPrefix, FilePerThread";
  }

  // Initialize worker part of shared resources (geometry, physics)
  G4WorkerThread::BuildGeometryAndPhysicsVector();

  // Create worker run manager
  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  if(!kernel) kernel = new G4WorkerRunManagerKernel();

  // Set the geometry for the worker, share from master
  DDDWorld::WorkerSetAsWorld(runManagerMaster.world().GetWorldVolumeForWorker());

  // we need the track manager now
  m_tls->trackManager.reset(new SimTrackManager());

  // Get DDCompactView, or would it be better to get the object from
  // runManagerMaster instead of EventSetup in here?
  edm::ESTransientHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);

  // setup the magnetic field
  if (m_pUseMagneticField)
    {
      const GlobalPoint g(0.,0.,0.);

      edm::ESHandle<MagneticField> pMF;
      es.get<IdealMagneticFieldRecord>().get(pMF);

      m_tls->fieldBuilder.reset(new sim::FieldBuilder(pMF.product(), m_pField));
      G4TransportationManager * tM =
	G4TransportationManager::GetTransportationManager();
      m_tls->fieldBuilder->build( tM->GetFieldManager(),
                                  tM->GetPropagatorInField(),
                                  runManagerMaster.chordFinderSetterForWorker());
    }


  // attach sensitive detector
  AttachSD attach;
  std::pair< std::vector<SensitiveTkDetector*>,
    std::vector<SensitiveCaloDetector*> > sensDets =
    attach.create(runManagerMaster.world(),
                  (*pDD),
                  runManagerMaster.catalog(),
                  m_p,
                  m_tls->trackManager.get(),
                  m_tls->registry);

  m_tls->sensTkDets.swap(sensDets.first);
  m_tls->sensCaloDets.swap(sensDets.second);

  edm::LogInfo("SimG4CoreApplication")
    << " RunManagerMTWorker: Sensitive Detector "
    << "building finished; found "
    << m_tls->sensTkDets.size()
    << " Tk type Producers, and "
    << m_tls->sensCaloDets.size()
    << " Calo type producers ";

  // Set the physics list for the worker, share from master
  PhysicsList *physicsList = runManagerMaster.physicsListForWorker();

  edm::LogInfo("SimG4CoreApplication") 
    << "RunManagerMTWorker: start initialisation of PhysicsList for a thread";

  physicsList->InitializeWorker();
  kernel->SetPhysics(physicsList);
  kernel->InitializePhysics();

  const bool kernelInit = kernel->RunInitialization();
  if(!kernelInit)
    throw SimG4Exception("G4WorkerRunManagerKernel initialization failed");

  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_tls->registry.beginOfJobSignal_(&aBeginOfJob);

  initializeUserActions();
  /*
  for(const std::string& command: runManagerMaster.G4Commands()) {
    edm::LogInfo("SimG4CoreApplication") << "RunManagerMTWorker:: Requests UI: "
                                         << command;
    G4UImanager::GetUIpointer()->ApplyCommand(command);
  }
  */
}

void RunManagerMTWorker::initializeUserActions() {
  m_tls->runInterface.reset(new SimRunInterface(this, false));
  m_tls->userRunAction.reset(new RunAction(m_pRunAction, m_tls->runInterface.get()));
  m_tls->userRunAction->SetMaster(false);
  Connect(m_tls->userRunAction.get());

  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  G4EventManager * eventManager = kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  EventAction * userEventAction =
    new EventAction(m_pEventAction, m_tls->runInterface.get(), m_tls->trackManager.get());
  Connect(userEventAction);
  eventManager->SetUserAction(userEventAction);

  TrackingAction* userTrackingAction =
    new TrackingAction(userEventAction,m_pTrackingAction);
  Connect(userTrackingAction);
  eventManager->SetUserAction(userTrackingAction);

  SteppingAction* userSteppingAction =
    new SteppingAction(userEventAction,m_pSteppingAction); 
  Connect(userSteppingAction);
  eventManager->SetUserAction(userSteppingAction);

  eventManager->SetUserAction(new StackingAction(userTrackingAction,
						 m_pStackingAction));

}

void  RunManagerMTWorker::Connect(RunAction* runAction)
{
  runAction->m_beginOfRunSignal.connect(m_tls->registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_tls->registry.endOfRunSignal_);
}

void  RunManagerMTWorker::Connect(EventAction* eventAction)
{
  eventAction->m_beginOfEventSignal.connect(m_tls->registry.beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_tls->registry.endOfEventSignal_);
}

void  RunManagerMTWorker::Connect(TrackingAction* trackingAction)
{
  trackingAction->m_beginOfTrackSignal.connect(m_tls->registry.beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_tls->registry.endOfTrackSignal_);
}

void  RunManagerMTWorker::Connect(SteppingAction* steppingAction)
{
  steppingAction->m_g4StepSignal.connect(m_tls->registry.g4StepSignal_);
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
std::vector<std::shared_ptr<SimProducer> > RunManagerMTWorker::producers() {
  initializeTLS();
  return m_tls->producers;
}


void RunManagerMTWorker::initializeRun() {
  m_tls->currentRun.reset(new G4Run());
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (m_tls->userRunAction) { m_tls->userRunAction->BeginOfRunAction(m_tls->currentRun.get()); }
}

void RunManagerMTWorker::terminateRun() {
  if(m_tls && m_tls->userRunAction) {
    m_tls->userRunAction->EndOfRunAction(m_tls->currentRun.get());
    m_tls->userRunAction.reset();
  }

  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  if(kernel && m_tls && !m_tls->runTerminated) {
    m_tls->currentEvent.reset();
    m_simEvent.reset();
    kernel->RunTermination();
    m_tls->runTerminated = true;
  }
}

void RunManagerMTWorker::produce(const edm::Event& inpevt, const edm::EventSetup& es, const RunManagerMT& runManagerMaster) {
  // The initialization and begin/end run is a bit convoluted due to
  // - Geant4 deals per-thread
  // - OscarMTProducer deals per-stream
  // and framework/TBB is free to schedule work in streams to the
  // threads as it likes.
  //
  // We have to do the per-thread initialization, and per-thread
  // per-run initialization here by ourselves. 
  if(!(m_tls && m_tls->threadInitialized)) {
    LogDebug("SimG4CoreApplication") << "RunManagerMTWorker::produce(): stream " << inpevt.streamID() << " thread " << getThreadIndex() << " initializing";
    initializeThread(runManagerMaster, es);
    m_tls->threadInitialized = true;
  }
  // Initialize run
  if(inpevt.id().run() != m_tls->currentRunNumber) {
    if(m_tls->currentRunNumber != 0 && !m_tls->runTerminated) {
      // If previous run in this thread was not terminated via endRun() call, terminate it now
      terminateRun();
    }
    initializeRun();
    m_tls->currentRunNumber = inpevt.id().run();
  }
  m_tls->runInterface->setRunManagerMTWorker(this); // For UserActions


  m_tls->currentEvent.reset(generateEvent(inpevt));

  m_simEvent.reset(new G4SimEvent());
  m_simEvent->hepEvent(m_generator.genEvent());
  m_simEvent->weight(m_generator.eventWeight());
  if (m_generator.genVertex() !=0 ) {
    auto genVertex = m_generator.genVertex();
    m_simEvent->collisionPoint(
      math::XYZTLorentzVectorD(genVertex->x()/centimeter,
			       genVertex->y()/centimeter,
			       genVertex->z()/centimeter,
			       genVertex->t()/second));
  }
  if (m_tls->currentEvent->GetNumberOfPrimaryVertex()==0) {
    edm::LogError("SimG4CoreApplication") 
      << " RunManagerMT::produce event " << inpevt.id().event()
      << " with no G4PrimaryVertices \n  Aborting Run" ;
       
    abortRun(false);
  } else {
    G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
    if(!kernel) {
      std::stringstream ss;
      ss << "No G4WorkerRunManagerKernel yet for thread index" << getThreadIndex() << ", id " << std::hex << std::this_thread::get_id();
      throw SimG4Exception(ss.str());
    }
    kernel->GetEventManager()->ProcessOneEvent(m_tls->currentEvent.get());
  }
    
  edm::LogInfo("SimG4CoreApplication")
    << " RunManagerMTWorker: saved : Event  " << inpevt.id().event() 
    << " stream id " << inpevt.streamID()
    << " thread index " << getThreadIndex()
    << " of weight " << m_simEvent->weight()
    << " with " << m_simEvent->nTracks() << " tracks and " 
    << m_simEvent->nVertices()
    << " vertices, generated by " << m_simEvent->nGenParts() << " particles ";
}

void RunManagerMTWorker::abortEvent() {
  if(m_tls->runTerminated) { return; }
  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  G4Track* t = kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill) ;

  // CMS-specific act
  //
  TrackingAction* uta =
    static_cast<TrackingAction *>(kernel->GetEventManager()->GetUserTrackingAction());
  uta->PostUserTrackingAction(t) ;

  m_tls->currentEvent->SetEventAborted();

  // do NOT call this method for now
  // because it'll set abortRequested=true (withing G4EventManager)
  // this will make Geant4, in the event *next* after the aborted one
  // NOT to get the primary, thus there's NOTHING to trace, and it goes
  // to the end of G4Event::DoProcessing(G4Event*), where abortRequested
  // will be reset to true again
  //
  //kernel->GetEventManager()->AbortCurrentEvent();
  //
  // instead, mimic what it does, except (re)setting abortRequested
  //
  kernel->GetEventManager()->GetStackManager()->clear() ;
  kernel->GetEventManager()->GetTrackingManager()->EventAborted() ;

  G4StateManager* stateManager = G4StateManager::GetStateManager();
  stateManager->SetNewState(G4State_GeomClosed);
}

void RunManagerMTWorker::abortRun(bool softAbort) {
  if (!softAbort) { abortEvent(); }
  m_tls->currentRun.reset();
  terminateRun();
}

G4Event * RunManagerMTWorker::generateEvent(const edm::Event& inpevt) {
  m_tls->currentEvent.reset();
  m_simEvent.reset();

  // 64 bits event ID in CMSSW converted into Geant4 event ID
  G4int evtid = (G4int)inpevt.id().event();
  G4Event * evt = new G4Event(evtid);

  edm::Handle<edm::HepMCProduct> HepMCEvt;

  inpevt.getByToken(m_InToken, HepMCEvt);

  m_generator.setGenEvent(HepMCEvt->GetEvent());

  // required to reset the GenParticle Id for particles transported
  // along the beam pipe
  // to their original value for SimTrack creation
  resetGenParticleId( inpevt );

  if (!m_nonBeam) 
    {
      m_generator.HepMC2G4(HepMCEvt->GetEvent(),evt);
    }
  else 
    {
      m_generator.nonBeamEvent2G4(HepMCEvt->GetEvent(),evt);
    }

  return evt;
}

void RunManagerMTWorker::resetGenParticleId(const edm::Event& inpevt)
{
  edm::Handle<edm::LHCTransportLinkContainer> theLHCTlink;
  inpevt.getByToken( m_theLHCTlinkToken, theLHCTlink );
  if ( theLHCTlink.isValid() ) {
    m_tls->trackManager->setLHCTransportLink( theLHCTlink.product() );
  }
}
