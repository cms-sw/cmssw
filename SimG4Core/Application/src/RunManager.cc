#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/PrimaryTransformer.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/G4RegionReporter.h"
#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/Physics/interface/DDG4ProductionCuts.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/ChordFinderSetter.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

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

#include "G4GDMLParser.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

static
void createWatchers(const edm::ParameterSet& iP,
		    SimActivityRegistry& iReg,
		    std::vector<std::shared_ptr<SimWatcher> >& oWatchers,
		    std::vector<std::shared_ptr<SimProducer> >& oProds
   )
{
  using namespace std;
  using namespace edm;

  vector<ParameterSet> watchers = iP.getParameter<vector<ParameterSet> >("Watchers");
  
  for(vector<ParameterSet>::iterator itWatcher = watchers.begin();
      itWatcher != watchers.end();
      ++itWatcher) {
    std::shared_ptr<SimWatcherMakerBase> maker( 
      SimWatcherFactory::get()->create
      (itWatcher->getParameter<std::string> ("type")) );
    if(maker.get()==nullptr) {
      throw edm::Exception(edm::errors::Configuration)
	<< "Unable to find the requested Watcher";
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

RunManager::RunManager(edm::ParameterSet const & p, edm::ConsumesCollector&& iC) 
  :   m_generator(new Generator(p.getParameter<edm::ParameterSet>("Generator"))),
      m_HepMC(iC.consumes<edm::HepMCProduct>(p.getParameter<edm::ParameterSet>("Generator").getParameter<std::string>("HepMCProductLabel"))),
      m_LHCtr(iC.consumes<edm::LHCTransportLinkContainer>(p.getParameter<edm::InputTag>("theLHCTlinkTag"))),
      m_nonBeam(p.getParameter<bool>("NonBeamEvent")), 
      m_primaryTransformer(nullptr), 
      m_managerInitialized(false), 
      m_runInitialized(false), m_runTerminated(false), m_runAborted(false),
      firstRun(true),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_currentRun(0), m_currentEvent(0), m_simEvent(0), 
      m_PhysicsTablesDir(p.getParameter<std::string>("PhysicsTablesDirectory")),
      m_StorePhysicsTables(p.getParameter<bool>("StorePhysicsTables")),
      m_RestorePhysicsTables(p.getParameter<bool>("RestorePhysicsTables")),
      m_EvtMgrVerbosity(p.getUntrackedParameter<int>("G4EventManagerVerbosity",0)),
      m_pField(p.getParameter<edm::ParameterSet>("MagneticField")),
      m_pGenerator(p.getParameter<edm::ParameterSet>("Generator")),
      m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
      m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),      
      m_pEventAction(p.getParameter<edm::ParameterSet>("EventAction")),
      m_pStackingAction(p.getParameter<edm::ParameterSet>("StackingAction")),
      m_pTrackingAction(p.getParameter<edm::ParameterSet>("TrackingAction")),
      m_pSteppingAction(p.getParameter<edm::ParameterSet>("SteppingAction")),
      m_g4overlap(p.getParameter<edm::ParameterSet>("G4CheckOverlap")),
      m_G4Commands(p.getParameter<std::vector<std::string> >("G4Commands")),
      m_p(p), m_fieldBuilder(nullptr), m_chordFinderSetter(nullptr)
{    
  m_UIsession.reset(new CustomUIsession());
  m_kernel = new G4RunManagerKernel();

  m_check = p.getUntrackedParameter<bool>("CheckOverlap",false);
  m_WriteFile = p.getUntrackedParameter<std::string>("FileNameGDML","");
  m_FieldFile = p.getUntrackedParameter<std::string>("FileNameField","");
  m_RegionFile = p.getUntrackedParameter<std::string>("FileNameRegions","");

  m_userRunAction = nullptr;
  m_runInterface = nullptr;

  //Look for an outside SimActivityRegistry
  // this is used by the visualization code
  edm::Service<SimActivityRegistry> otherRegistry;
  if(otherRegistry){
    m_registry.connect(*otherRegistry);
  }
  m_sVerbose.reset(nullptr);

  std::vector<edm::ParameterSet> watchers 
    = p.getParameter<std::vector<edm::ParameterSet> >("Watchers");
  m_hasWatchers = (watchers.empty()) ? false : true;

  if(m_hasWatchers) {
    createWatchers(m_p, m_registry, m_watchers, m_producers);
  }
}

RunManager::~RunManager() 
{ 
  if (!m_runTerminated) { terminateRun(); }
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  G4GeometryManager::GetInstance()->OpenGeometry();
  //   if (m_kernel!=0) delete m_kernel; 
  delete m_runInterface;
  delete m_generator;
}

void RunManager::initG4(const edm::EventSetup & es)
{
  bool geomChanged = idealGeomRcdWatcher_.check(es);
  if (geomChanged && (!firstRun)) {
    throw cms::Exception("BadConfig") 
      << "[SimG4Core RunManager]\n"
      << "The Geometry configuration is changed during the job execution\n"
      << "this is not allowed, the geometry must stay unchanged\n";
  }
  if (m_pUseMagneticField) {
    bool magChanged = idealMagRcdWatcher_.check(es);
    if (magChanged && (!firstRun)) {
      throw edm::Exception(edm::errors::Configuration) 
	<< "[SimG4Core RunManager]\n"
	<< "The MagneticField configuration is changed during the job execution\n"
	<< "this is not allowed, the MagneticField must stay unchanged\n";
    }
  }

  if (m_managerInitialized) return;
  
  // DDDWorld: get the DDCV from the ES and use it to build the World
  edm::ESTransientHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);
   
  G4LogicalVolumeToDDLogicalPartMap map_;
  SensitiveDetectorCatalog catalog_;
  const DDDWorld * world = new DDDWorld(&(*pDD), map_, catalog_, false);
  m_registry.dddWorldSignal_(world);

  if (m_pUseMagneticField)
    {
      // setup the magnetic field
      edm::ESHandle<MagneticField> pMF;
      es.get<IdealMagneticFieldRecord>().get(pMF);
      const GlobalPoint g(0.,0.,0.);

      m_chordFinderSetter = new sim::ChordFinderSetter();
      m_fieldBuilder = new sim::FieldBuilder(&(*pMF), m_pField);
      G4TransportationManager * tM = 
	G4TransportationManager::GetTransportationManager();
      m_fieldBuilder->build( tM->GetFieldManager(),
			     tM->GetPropagatorInField(),
                             m_chordFinderSetter);
      if("" != m_FieldFile) { 
	DumpMagneticField(tM->GetFieldManager()->GetDetectorField()); 
      }
    }

  // we need the track manager now
  m_trackManager = std::unique_ptr<SimTrackManager>(new SimTrackManager);

  // attach sensitive detector
  m_attach = new AttachSD;
  
  std::pair< std::vector<SensitiveTkDetector*>,
    std::vector<SensitiveCaloDetector*> > sensDets = 
    m_attach->create(*world,(*pDD),catalog_,m_p,m_trackManager.get(),
		     m_registry);
      
  m_sensTkDets.swap(sensDets.first);
  m_sensCaloDets.swap(sensDets.second);
     
  edm::LogInfo("SimG4CoreApplication") 
    << " RunManager: Sensitive Detector "
    << "building finished; found " 
    << m_sensTkDets.size()
    << " Tk type Producers, and " 
    << m_sensCaloDets.size() 
    << " Calo type producers ";

  edm::ESHandle<HepPDT::ParticleDataTable> fTable;
  es.get<PDTRecord>().get(fTable);
  const HepPDT::ParticleDataTable *fPDGTable = &(*fTable);

  m_primaryTransformer = new PrimaryTransformer();

  std::unique_ptr<PhysicsListMakerBase> 
    physicsMaker(PhysicsListFactory::get()->create(
      m_pPhysics.getParameter<std::string> ("type")));
  if (physicsMaker.get()==nullptr) {
    throw edm::Exception(edm::errors::Configuration)
      << "Unable to find the Physics list requested";
  }
  m_physicsList = 
    physicsMaker->make(map_,fPDGTable,m_chordFinderSetter,m_pPhysics,m_registry);

  PhysicsList* phys = m_physicsList.get(); 
  if (phys==nullptr) { 
    throw edm::Exception(edm::errors::Configuration)
      << "Physics list construction failed!"; 
  }

  // adding GFlash, Russian Roulette for eletrons and gamma, 
  // step limiters on top of any Physics Lists
  phys->RegisterPhysics(new ParametrisedEMPhysics("EMoptions",m_pPhysics));

  m_physicsList->ResetStoredInAscii();
  std::string tableDir = m_PhysicsTablesDir;
  if (m_RestorePhysicsTables) {
    m_physicsList->SetPhysicsTableRetrieved(tableDir);
  } 
  edm::LogInfo("SimG4CoreApplication") 
    << "RunManager: start initialisation of PhysicsList";

  int verb = std::max(m_pPhysics.getUntrackedParameter<int>("Verbosity",0),
		      m_p.getParameter<int>("SteppingVerbosity"));

  m_physicsList->SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*CLHEP::cm);
  m_physicsList->SetCutsWithDefault();
  m_prodCuts.reset(new DDG4ProductionCuts(map_, verb, m_pPhysics));	
  m_prodCuts->update();

  m_kernel->SetPhysics(phys);
  m_kernel->InitializePhysics();

  if (m_kernel->RunInitialization()) { m_managerInitialized = true; }
  else { 
    throw edm::Exception(edm::errors::LogicError)
      << "G4RunManagerKernel initialization failed!"; 
  }
  
  if (m_StorePhysicsTables)
    {
      std::ostringstream dir;
      dir << tableDir << '\0';
      std::string cmd = std::string("/control/shell mkdir -p ")+tableDir;
      if (!std::ifstream(dir.str().c_str(), std::ios::in))
        G4UImanager::GetUIpointer()->ApplyCommand(cmd);
      m_physicsList->StorePhysicsTable(tableDir);
    }
  
  //tell all interesting parties that we are beginning the job
  BeginOfJob aBeginOfJob(&es);
  m_registry.beginOfJobSignal_(&aBeginOfJob);
  
  G4int sv = m_p.getParameter<int>("SteppingVerbosity");
  G4double elim = m_p.getParameter<double>("StepVerboseThreshold")*CLHEP::GeV;
  std::vector<int> ve = m_p.getParameter<std::vector<int> >("VerboseEvents");
  std::vector<int> vn = m_p.getParameter<std::vector<int> >("VertexNumber");
  std::vector<int> vt = m_p.getParameter<std::vector<int> >("VerboseTracks");

  if(sv > 0) {
    m_sVerbose.reset(new CMSSteppingVerbose(sv, elim, ve, vn, vt));
  }
  initializeUserActions();
  
  if(0 < m_G4Commands.size()) {
    G4cout << "RunManager: Requested UI commands: " << G4endl;
    for (unsigned it=0; it<m_G4Commands.size(); ++it) {
      G4cout << "    " << m_G4Commands[it] << G4endl;
      G4UImanager::GetUIpointer()->ApplyCommand(m_G4Commands[it]);
    }
  }

  if("" != m_WriteFile) {
    G4GDMLParser gdml;
    gdml.Write(m_WriteFile, world->GetWorldVolume(), true);
  }

  if("" != m_RegionFile) {
    G4RegionReporter rrep;
    rrep.ReportRegions(m_RegionFile);
  }

  if(m_check) { G4CheckOverlap check(m_g4overlap); }

  // If the Geant4 particle table is needed, decomment the lines below
  //
  //  G4cout << "Output of G4ParticleTable DumpTable:" << G4endl;
  //  G4ParticleTable::GetParticleTable()->DumpTable("ALL");
  
  initializeRun();
  firstRun= false;
}

void RunManager::stopG4()
{
  G4StateManager::GetStateManager()->SetNewState(G4State_Quit);
  if (!m_runTerminated) { terminateRun(); }
}

void RunManager::produce(edm::Event& inpevt, const edm::EventSetup & es)
{
  m_currentEvent = generateEvent(inpevt);
  m_simEvent = new G4SimEvent;
  m_simEvent->hepEvent(m_generator->genEvent());
  m_simEvent->weight(m_generator->eventWeight());
  if (m_generator->genVertex() !=0 ) {
    m_simEvent->collisionPoint(
      math::XYZTLorentzVectorD(m_generator->genVertex()->x()/centimeter,
			       m_generator->genVertex()->y()/centimeter,
			       m_generator->genVertex()->z()/centimeter,
			       m_generator->genVertex()->t()/second));
  }
  if (m_currentEvent->GetNumberOfPrimaryVertex()==0) {
    std::stringstream ss;
    ss << " RunManager::produce(): event " << inpevt.id().event()
       << " with no G4PrimaryVertices\n" ;
    throw SimG4Exception(ss.str());
       
    abortRun(false);
  } else {
    edm::LogInfo("SimG4CoreApplication") 
      << "RunManager::produce: start Event " << inpevt.id().event() 
      << " of weight " << m_simEvent->weight()
      << " with " << m_simEvent->nTracks() << " tracks and " 
      << m_simEvent->nVertices()
      << " vertices, generated by " << m_simEvent->nGenParts() << " particles ";

    m_kernel->GetEventManager()->ProcessOneEvent(m_currentEvent);

    edm::LogInfo("SimG4CoreApplication")
      << " RunManager::produce: ended Event " << inpevt.id().event(); 
  }    
}
 
G4Event * RunManager::generateEvent(edm::Event & inpevt)
{                       
  if (m_currentEvent!=0) { delete m_currentEvent; }
  m_currentEvent = 0;
  if (m_simEvent!=0) { delete m_simEvent; }
  m_simEvent = 0;

  // 64 bits event ID in CMSSW converted into Geant4 event ID
  G4int evtid = (G4int)inpevt.id().event();
  G4Event * evt = new G4Event(evtid);
  
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  
  inpevt.getByToken( m_HepMC, HepMCEvt ) ;
  
  m_generator->setGenEvent(HepMCEvt->GetEvent());

  // required to reset the GenParticle Id for particles transported 
  // along the beam pipe
  // to their original value for SimTrack creation
  resetGenParticleId( inpevt );

  if (!m_nonBeam) 
    {
      m_generator->HepMC2G4(HepMCEvt->GetEvent(),evt);
    }
  else 
    {
      m_generator->nonBeamEvent2G4(HepMCEvt->GetEvent(),evt);
    }
 
  return evt;
}

void RunManager::abortEvent()
{
  if (m_runTerminated) { return; }
  G4Track* t =
    m_kernel->GetEventManager()->GetTrackingManager()->GetTrack();
  t->SetTrackStatus(fStopAndKill) ;
     
  // CMS-specific act
  //
  TrackingAction* uta =
    (TrackingAction*)m_kernel->GetEventManager()->GetUserTrackingAction() ;
  uta->PostUserTrackingAction(t) ;

  m_currentEvent->SetEventAborted();    
  m_kernel->GetEventManager()->GetStackManager()->clear() ;
  m_kernel->GetEventManager()->GetTrackingManager()->EventAborted() ;
     
  G4StateManager* stateManager = G4StateManager::GetStateManager();
  stateManager->SetNewState(G4State_GeomClosed);
}

void RunManager::initializeUserActions()
{
  m_runInterface = new SimRunInterface(this, false);

  m_userRunAction = new RunAction(m_pRunAction, m_runInterface, true);
  Connect(m_userRunAction);

  G4EventManager * eventManager = m_kernel->GetEventManager();
  eventManager->SetVerboseLevel(m_EvtMgrVerbosity);

  if (m_generator!=nullptr) {
    EventAction * userEventAction = 
      new EventAction(m_pEventAction, m_runInterface, m_trackManager.get(),
		      m_sVerbose.get());
    Connect(userEventAction);
    eventManager->SetUserAction(userEventAction);

    TrackingAction* userTrackingAction = 
      new TrackingAction(userEventAction,m_pTrackingAction,m_sVerbose.get());
    Connect(userTrackingAction);
    eventManager->SetUserAction(userTrackingAction);
	
    SteppingAction* userSteppingAction = 
      new SteppingAction(userEventAction,m_pSteppingAction,m_sVerbose.get(),m_hasWatchers); 
    Connect(userSteppingAction);
    eventManager->SetUserAction(userSteppingAction);

    eventManager->SetUserAction(new StackingAction(userTrackingAction, 
						   m_pStackingAction,m_sVerbose.get()));

  } else {
    edm::LogWarning("SimG4CoreApplication") << " RunManager: WARNING : "
					    << "No generator; initialized "
					    << "only RunAction!";
  }
}

void RunManager::initializeRun()
{
  m_runInitialized = false;
  if (m_currentRun==nullptr) { m_currentRun = new G4Run(); }
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);
  if (m_userRunAction!=nullptr) { m_userRunAction->BeginOfRunAction(m_currentRun); }
  m_runAborted = false;
  m_runInitialized = true;
}
 
void RunManager::terminateRun()
{
  if(m_runTerminated) { return; }
  if (m_userRunAction!=nullptr) {
    m_userRunAction->EndOfRunAction(m_currentRun);
    delete m_userRunAction; 
    m_userRunAction = nullptr;
  }
  delete m_currentEvent;
  m_currentEvent = nullptr;
  delete m_simEvent;
  m_simEvent = nullptr;
  if(m_kernel != nullptr) { m_kernel->RunTermination(); }
  m_runInitialized = false;
  m_runTerminated = true;  
}

void RunManager::abortRun(bool softAbort)
{
  if(m_runAborted) { return; }
  if (!softAbort) { abortEvent(); }
  if (m_currentRun!=0) { delete m_currentRun; m_currentRun = 0; }
  terminateRun();
  m_runAborted = true;
}

void RunManager::resetGenParticleId( edm::Event& inpevt ) 
{
  edm::Handle<edm::LHCTransportLinkContainer> theLHCTlink;
  inpevt.getByToken( m_LHCtr, theLHCTlink );
  if ( theLHCTlink.isValid() ) {
    m_trackManager->setLHCTransportLink( theLHCTlink.product() );
  }
}

SimTrackManager* RunManager::GetSimTrackManager()
{
  return m_trackManager.get();
}

void  RunManager::Connect(RunAction* runAction)
{
  runAction->m_beginOfRunSignal.connect(m_registry.beginOfRunSignal_);
  runAction->m_endOfRunSignal.connect(m_registry.endOfRunSignal_);
}

void  RunManager::Connect(EventAction* eventAction)
{
  eventAction->m_beginOfEventSignal.connect(m_registry.beginOfEventSignal_);
  eventAction->m_endOfEventSignal.connect(m_registry.endOfEventSignal_);
}

void  RunManager::Connect(TrackingAction* trackingAction)
{
  trackingAction->m_beginOfTrackSignal.connect(m_registry.beginOfTrackSignal_);
  trackingAction->m_endOfTrackSignal.connect(m_registry.endOfTrackSignal_);
}

void  RunManager::Connect(SteppingAction* steppingAction)
{
  steppingAction->m_g4StepSignal.connect(m_registry.g4StepSignal_);
}

void RunManager::DumpMagneticField(const G4Field* field) const
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
