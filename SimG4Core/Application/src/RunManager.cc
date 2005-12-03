#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "G4StateManager.hh"
#include "G4ApplicationState.hh"
#include "G4RunManagerKernel.hh"
#include "G4UImanager.hh"

#include "G4EventManager.hh"
#include "G4Run.hh"
#include "G4Event.hh"

#include "CLHEP/Random/JamesRandom.h"
 
#include "Randomize.hh"

#include <iostream>
#include <memory>
#include <strstream>
#include <fstream>

static
void createWatchers(const edm::ParameterSet& iP,
		    SimActivityRegistry& iReg,
		    std::vector<boost::shared_ptr<SimWatcher> >& oWatchers,
		    std::vector<boost::shared_ptr<SimProducer> >& oProds
   )
{
  using namespace std;
  using namespace edm;
  vector<ParameterSet> watchers;
  try {
    watchers = iP.getParameter<vector<ParameterSet> >("Watchers");
  } catch( edm::Exception) {
  }
  
  for(vector<ParameterSet>::iterator itWatcher = watchers.begin();
      itWatcher != watchers.end();
      ++itWatcher) {
    std::auto_ptr<SimWatcherMakerBase> maker( 
      SimWatcherFactory::get()->create
      (itWatcher->getParameter<std::string> ("type")) );
    if(maker.get()==0) {
      throw SimG4Exception("Unable to find the requested Watcher");
    }
    
    boost::shared_ptr<SimWatcher> watcherTemp;
    boost::shared_ptr<SimProducer> producerTemp;
    maker->make(*itWatcher,iReg,watcherTemp,producerTemp);
    oWatchers.push_back(watcherTemp);
    if(producerTemp) {
       oProds.push_back(producerTemp);
    }
  }
}

RunManager * RunManager::me = 0;
RunManager * RunManager::init(edm::ParameterSet const & p)
{
    if (me != 0) abort();
    me = new RunManager(p);
    return me;
}

RunManager * RunManager::instance() 
{
    if (me==0) abort();
    return me;
}

RunManager::RunManager(edm::ParameterSet const & p) 
  :   m_generator(0), m_primaryTransformer(0), m_engine(0), m_managerInitialized(false), 
      m_geometryInitialized(true), m_physicsInitialized(true),
      m_runInitialized(false), m_runTerminated(false), m_runAborted(false),
      m_currentRun(0), m_currentEvent(0), m_simEvent(0), 
      m_rndmStore(p.getParameter<bool>("StoreRndmSeeds")),
      m_rndmRestore(p.getParameter<bool>("RestoreRndmSeeds")),
      m_PhysicsTablesDir(p.getParameter<std::string>("PhysicsTablesDirectory")),
      m_StorePhysicsTables(p.getParameter<bool>("StorePhysicsTables")),
      m_RestorePhysicsTables(p.getParameter<bool>("RestorePhysicsTables")),
      m_EvtMgrVerbosity(p.getParameter<int>("G4EventManagerVerbosity")),
      m_Override(p.getParameter<bool>("OverrideUserStackingAction")),
      m_RunNumber(p.getParameter<int>("RunNumber")),
      m_pGenerator(p.getParameter<edm::ParameterSet>("Generator")),
      m_pVertexGenerator(p.getParameter<edm::ParameterSet>("VertexGenerator")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),      
      m_pEventAction(p.getParameter<edm::ParameterSet>("EventAction")),
      m_pTrackingAction(p.getParameter<edm::ParameterSet>("TrackingAction")),
      m_pSteppingAction(p.getParameter<edm::ParameterSet>("SteppingAction")),
      m_p(p)
{    
    m_kernel = G4RunManagerKernel::GetRunManagerKernel();
    if (m_kernel==0) m_kernel = new G4RunManagerKernel();
    m_engine= dynamic_cast<HepJamesRandom*>(HepRandom::getTheEngine());
    std::cout << " Run Manager constructed " << std::endl;

    //Look for an outside SimActivityRegistry
    // this is used by the visualization code
    edm::Service<SimActivityRegistry> otherRegistry;
    if(otherRegistry){
       m_registry.connect(*otherRegistry);
    }

    createWatchers(m_p, m_registry, m_watchers, m_producers);
}

RunManager::~RunManager() 
{ 
    if (m_kernel!=0) delete m_kernel; 
}

void RunManager::initG4(const edm::EventSetup & es)
{
    if (m_managerInitialized) return;

    // DDDWorld: get the DDCV from the ES and use it to build the World
    edm::ESHandle<DDCompactView> pDD;
    es.get<IdealGeometryRecord>().get(pDD);
   
    //QUESTION: Who deletes this?
    const DDDWorld * world = new DDDWorld(&(*pDD));
    //ANSWER: Unfortunately we cannot; G4 deletes all the volumes for now
    //        but I'll check again
    m_registry.dddWorldSignal_(world);

    //QUESTION: Are the following two lines still needed?
    //ANSWER: No!

    //we need the track manager now
    m_trackManager = std::auto_ptr<SimTrackManager>(new SimTrackManager);

    m_attach = new AttachSD;
    {
      std::pair< std::vector<SensitiveTkDetector*>,
	std::vector<SensitiveCaloDetector*> > sensDets = m_attach->create(*world,(*pDD),m_p,m_trackManager.get(),m_registry);
      
      m_sensTkDets.swap(sensDets.first);
      m_sensCaloDets.swap(sensDets.second);
    }

    std::cout << " Sensitive Detector building finished; found " << m_sensTkDets.size()
	      << " Tk type Producers, and " << m_sensCaloDets.size() 
	      << " Calo type producers " << std::endl;

    std::auto_ptr<EventVertexGeneratorMakerBase> vertexGeneratorMaker(
      EventVertexGeneratorFactory::get()->create
      (m_pVertexGenerator.getParameter<std::string> ("type")) );
    if(vertexGeneratorMaker.get()==0) {
      throw SimG4Exception("Unable to find the event vertex generator requested");
    }
    m_eventVertexGenerator = vertexGeneratorMaker->make(m_pVertexGenerator,m_registry);
    if (m_eventVertexGenerator.get()==0) throw SimG4Exception("EventVertexGenerator construction failed!");

    m_generator = new Generator(m_pGenerator);
    m_primaryTransformer = new PrimaryTransformer();
    
    std::auto_ptr<PhysicsListMakerBase> physicsMaker( 
      PhysicsListFactory::get()->create
      (m_pPhysics.getParameter<std::string> ("type")) );
    if(physicsMaker.get()==0) {
      throw SimG4Exception("Unable to find the Physics list requested");
    }
    m_physicsList = physicsMaker->make(m_pPhysics,m_registry);
    if (m_physicsList.get()==0) throw SimG4Exception("Physics list construction failed!");
    m_kernel->SetPhysics(m_physicsList.get());
    m_kernel->InitializePhysics();

    m_physicsList->ResetStoredInAscii();
    std::string tableDir = m_PhysicsTablesDir;
    if (m_RestorePhysicsTables) m_physicsList->SetPhysicsTableRetrieved(tableDir);
 
    if (m_kernel->RunInitialization()) m_managerInitialized = true;
    else throw SimG4Exception("G4RunManagerKernel initialization failed!");
     
    if (m_StorePhysicsTables)
    {
	std::ostrstream dir;
	dir << tableDir << '\0';
	std::string cmd = std::string("/control/shell mkdir -p ")+tableDir;
	if (!std::ifstream(dir.str(), std::ios::in))
	    G4UImanager::GetUIpointer()->ApplyCommand(cmd);
	m_physicsList->StorePhysicsTable(tableDir);
    }
 
    //tell all interesting parties that we are beginning the job
    BeginOfJob aBeginOfJob(&es);
    m_registry.beginOfJobSignal_(&aBeginOfJob);

    initializeUserActions();
    initializeRun();

}

void RunManager::produce(edm::Event& inpevt, const edm::EventSetup & es)
{
    static int i = 0;
    m_currentEvent = generateEvent(i,inpevt);
    i++;
    
    m_simEvent = new G4SimEvent;
    m_simEvent->hepEvent(m_generator->genEvent());
    m_simEvent->weight(m_generator->eventWeight());
    m_simEvent->collisionPoint(HepLorentzVector(m_generator->genVertex().vect()/centimeter,
                                                m_generator->genVertex().t()/second));
 
    if (m_currentEvent->GetNumberOfPrimaryVertex()==0)
    {
       std::cout << " RunManager::produce event " << i
            << " with no G4PrimaryVertices " << std::endl;
       std::cout << " Aborting run " << std::endl;
       abortRun(false);
    }
    else
        m_kernel->GetEventManager()->ProcessOneEvent(m_currentEvent);

    std::cout << " saved : Event  " << i << " of weight " << m_simEvent->weight()
         << " with " << m_simEvent->nTracks() << " tracks and " << m_simEvent->nVertices()
         << " vertices, generated by " << m_simEvent->nGenParts() << " particles " << std::endl;

}
 
G4Event * RunManager::generateEvent(int i, edm::Event& inpevt)
{                       

    if (m_currentEvent!=0) delete m_currentEvent;
    m_currentEvent = 0;
    if (m_simEvent!=0) delete m_simEvent;
    m_simEvent = 0;
    G4Event * e = new G4Event(i);
   
    if ( m_generator->genInputType() == "Internal" )
    {
       edm::Handle<edm::HepMCProduct> HepMCEvt ;
       inpevt.getByType( HepMCEvt ) ;
       if ( !HepMCEvt.isValid() )
       {
          std::cout << "Can NOT find HepMC Event " << std::endl;
          throw SimG4Exception("Unable to find HepMCProduct(HepMC::GenEvent) in edm::Event  ") ;
       }
       else
       {
          m_generator->setGenEvent( HepMCEvt->GetEvent() ) ;
          m_generator->HepMC2G4( HepMCEvt->GetEvent(), e ) ;
       }
    }
    else
    {    
       const HepMC::GenEvent * g = m_generator->generateEvent();
       m_generator->HepMC2G4(g,e);
    }
    
    return e;
}

void RunManager::abortEvent()
{
    m_currentEvent->SetEventAborted();
    m_kernel->GetEventManager()->AbortCurrentEvent();
}

void RunManager::initializeUserActions()
{
    RunAction* userRunAction = new RunAction(m_pRunAction);
    m_userRunAction = userRunAction;
    userRunAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
    userRunAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);

    G4EventManager * eventManager = m_kernel->GetEventManager();
    eventManager->SetVerboseLevel(m_EvtMgrVerbosity);
    if (m_generator!=0)
    {
        EventAction * userEventAction = new EventAction(m_pEventAction,m_trackManager.get());
	userEventAction->m_beginOfEventSignal.connect(m_registry.beginOfEventSignal_);
	userEventAction->m_endOfEventSignal.connect(m_registry.endOfEventSignal_);
        eventManager->SetUserAction(userEventAction);
        TrackingAction* userTrackingAction = new TrackingAction(userEventAction,m_pTrackingAction);
	userTrackingAction->m_beginOfTrackSignal.connect(m_registry.beginOfTrackSignal_);
	userTrackingAction->m_endOfTrackSignal.connect(m_registry.endOfTrackSignal_);
	eventManager->SetUserAction(userTrackingAction);
	
	SteppingAction* userSteppingAction = new SteppingAction(m_pSteppingAction); 
	userSteppingAction->m_g4StepSignal.connect(m_registry.g4StepSignal_);
        eventManager->SetUserAction(userSteppingAction);
        if (m_Override)
        {
            std::cout << " RunManager: user StackingAction overridden " 
		      << std::endl;
            eventManager->SetUserAction(new StackingAction);
        }
    }
    else std::cout << " WARNING: No generator; initialized only RunAction!" 
		   << std::endl;
}

void RunManager::initializeRun()
{
    m_runInitialized = false;
    if (m_currentRun==0) m_currentRun = new G4Run();
    m_currentRun->SetRunID(m_RunNumber);
    G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
    if (m_userRunAction!=0) m_userRunAction->BeginOfRunAction(m_currentRun);
    m_runAborted = false;
    if (m_rndmStore) runRNDMstore(m_RunNumber);
    if (m_rndmRestore) runRNDMrestore(m_RunNumber);
    m_runInitialized = true;
}
 
void RunManager::terminateRun()
{
    m_runTerminated = false;
    if (m_userRunAction!=0)
    {
        m_userRunAction->EndOfRunAction(m_currentRun);
        delete m_userRunAction; m_userRunAction = 0;
    }
    if (m_currentRun!=0) { delete m_currentRun; m_currentRun = 0; }
    if (m_kernel!=0) m_kernel->RunTermination();
    m_runInitialized = false;
    m_runTerminated = true;
}

void RunManager::abortRun(bool softAbort)
{
    m_runAborted = false;
    if (!softAbort) abortEvent();
    if (m_currentRun!=0) { delete m_currentRun; m_currentRun = 0; }
    m_runInitialized = false;
    m_runAborted = true;
}

void RunManager::runRNDMstore(int run)
{
    std::ostrstream dir;
    dir << "Run" << run << '\0';
    std::string cmd = std::string("/control/shell mkdir -p ")+dir.str();
    G4UImanager::GetUIpointer()->ApplyCommand(cmd);
    std::ostrstream os;
    os << "Run" << run << "/run" << run << ".rndm" << '\0';
    m_engine->saveStatus(os.str());
    std::cout << "Random number status saved in: " << os.str() << std::endl;
    m_engine->showStatus();
}
 
void RunManager::runRNDMrestore(int run)
{
    std::ostrstream os;
    os << "Run" << run << "/run" << run << ".rndm" << '\0';
    if (!std::ifstream(os.str(), std::ios::in))
    {
        std::cout << " rndm directory does not exist for run " << run << std::endl;
        return;
    }
    m_engine->restoreStatus(os.str());
    std::cout << "Random number status restored from: " << os.str() << std::endl;
    m_engine->showStatus();
}
 
void RunManager::eventRNDMstore(int run, int event)
{
    std::ostrstream os;
    os << "Run" << run << "/evt" << event << ".rndm" << '\0';
    m_engine->saveStatus(os.str());
    if (m_EvtMgrVerbosity>2)
    {
        std::cout << " random numbers saved in: " << os.str() << std::endl;
        m_engine->showStatus();
    }
}

void RunManager::eventRNDMrestore(int run, int event)
{
    std::ostrstream os;
    os << "Run" << run << "/evt" << event << ".rndm" << '\0';
    if (!std::ifstream(os.str(), std::ios::in))
    {
        std::cout << " rndm file does not exist for event " << event << std::endl;
        return;
    }
    m_engine->restoreStatus(os.str());
    if (m_EvtMgrVerbosity>2)
    {
        std::cout << "Random number status restored from: " << os.str() <<std:: endl;
        m_engine->showStatus();
    }
}
 
