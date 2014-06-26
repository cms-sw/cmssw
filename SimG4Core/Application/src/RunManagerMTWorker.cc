#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "G4Event.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4WorkerThread.hh"
#include "G4WorkerRunManagerKernel.hh"
#include "G4StateManager.hh"

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
}

thread_local bool RunManagerMTWorker::m_threadInitialized = false;
//thread_local std::unique_ptr<G4Run> RunManagerMTWorker::m_currentRun;
thread_local G4Run *RunManagerMTWorker::m_currentRun = nullptr;
thread_local RunAction *RunManagerMTWorker::m_userRunAction = nullptr;
thread_local SimRunInterface *RunManagerMTWorker::m_runInterface = nullptr;
thread_local SimTrackManager *RunManagerMTWorker::m_trackManager = nullptr;

RunManagerMTWorker::RunManagerMTWorker(const edm::ParameterSet& iConfig):
  m_generator(iConfig.getParameter<edm::ParameterSet>("Generator")),
  m_InTag(iConfig.getParameter<edm::ParameterSet>("Generator").getParameter<std::string>("HepMCProductLabel")),
  m_nonBeam(iConfig.getParameter<bool>("NonBeamEvent")),
  m_EvtMgrVerbosity(iConfig.getUntrackedParameter<int>("G4EventManagerVerbosity",0)),
  m_pRunAction(iConfig.getParameter<edm::ParameterSet>("RunAction")),
  m_pEventAction(iConfig.getParameter<edm::ParameterSet>("EventAction")),
  m_pStackingAction(iConfig.getParameter<edm::ParameterSet>("StackingAction")),
  m_pTrackingAction(iConfig.getParameter<edm::ParameterSet>("TrackingAction")),
  m_pSteppingAction(iConfig.getParameter<edm::ParameterSet>("SteppingAction"))
{

  edm::Service<SimActivityRegistry> otherRegistry;
  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  if(otherRegistry){
    m_registry.connect(*otherRegistry);
  }
}

RunManagerMTWorker::~RunManagerMTWorker() {}

void RunManagerMTWorker::beginRun(const RunManagerMT& runManagerMaster, const edm::EventSetup& es) {
  // Stream-specific beginRun
  // unfortunately does not work for per-thread initialization since framework does not guarantee to run them on differente threads...
  //edm::LogWarning("SimG4CoreApplication") << "RunManagerMTWorker::beginRun(): thread " << getThreadIndex();
}

void RunManagerMTWorker::initializeThread(const RunManagerMT& runManagerMaster) {
  // I guess everything initialized here should be in thread_local storage

  int thisID = getThreadIndex();

  // Initialize per-thread output
  G4Threading::G4SetThreadId( thisID );
  G4UImanager::GetUIpointer()->SetUpForAThread( thisID );

  // Initialize worker part of shared resources (geometry, physics)
  G4WorkerThread::BuildGeometryAndPhysicsVector();

  // Create worker run manager
  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  if(!kernel) kernel = new G4WorkerRunManagerKernel();

  // Set the geometry for the worker, share from master
  DDDWorld::SetAsWorld(runManagerMaster.world().GetWorldVolumeForWorker());

  // we need the track manager now
  m_trackManager = new SimTrackManager();

  // Set the physics list for the worker, share from master
  PhysicsList *physicsList = runManagerMaster.physicsListForWorker();
  physicsList->InitializeWorker();
  kernel->SetPhysics(physicsList);
  kernel->InitializePhysics();

  const bool kernelInit = kernel->RunInitialization();
  if(!kernelInit)
    throw SimG4Exception("G4WorkerRunManagerKernel initialization failed");

  initializeUserActions();

  // Initialize run
  //m_currentRun.reset(new G4Run());
  m_currentRun = new G4Run();
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
}

void RunManagerMTWorker::initializeUserActions() {
  m_runInterface = new SimRunInterface(this, false);

  m_userRunAction = new RunAction(m_pRunAction, m_runInterface);
  m_userRunAction->SetMaster(false);
  Connect(m_userRunAction);

  G4RunManagerKernel *kernel = G4WorkerRunManagerKernel::GetRunManagerKernel();
  G4EventManager * eventManager = kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  EventAction * userEventAction =
    new EventAction(m_pEventAction, m_runInterface, m_trackManager);
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

  eventManager->SetUserAction(new StackingAction(m_pStackingAction));

}

void RunManagerMTWorker::produce(const edm::Event& inpevt, const edm::EventSetup& es, const RunManagerMT& runManagerMaster) {

  if(!m_threadInitialized) {
    LogDebug("SimG4CoreApplication") << "RunManagerMTWorker::produce(): stream " << inpevt.streamID() << " thread " << getThreadIndex() << " initializing";
    initializeThread(runManagerMaster);
    m_threadInitialized = true;
  }
  m_runInterface->setRunManagerMTWorker(this); // For UserActions


  m_currentEvent.reset(generateEvent(inpevt));

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
  if (m_currentEvent->GetNumberOfPrimaryVertex()==0) {
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
    kernel->GetEventManager()->ProcessOneEvent(m_currentEvent.get());
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
}

void RunManagerMTWorker::abortRun(bool softAbort) {
}

G4Event * RunManagerMTWorker::generateEvent(const edm::Event& inpevt) {
  m_currentEvent.reset();
  m_simEvent.reset();

  G4Event * evt = new G4Event(inpevt.id().event());
  edm::Handle<edm::HepMCProduct> HepMCEvt;

  inpevt.getByLabel(m_InTag, HepMCEvt);

  m_generator.setGenEvent(HepMCEvt->GetEvent());

  // STUFF MISSING

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

