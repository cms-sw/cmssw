#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarProducer.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/Random.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

namespace {
    //
    // this machinery allows to set CLHEP static engine
    // to the one defined by RandomNumberGenerator service
    // at the beginning of an event, and reset it back to
    // "default-default" at the end of the event;
    // Dave D. has decided to implement it this way because
    // we don't know if there're other modules using CLHEP
    // static engine, thus we want to ensure that the one
    // we use for OscarProducer is unique to OscarProducer
    //
    class StaticRandomEngineSetUnset {
    public:
        StaticRandomEngineSetUnset();
        explicit StaticRandomEngineSetUnset(CLHEP::HepRandomEngine * engine);
        ~StaticRandomEngineSetUnset();
        CLHEP::HepRandomEngine* getEngine() const;
    private:
        CLHEP::HepRandomEngine* m_currentEngine;
        CLHEP::HepRandomEngine* m_previousEngine;
    };
}

OscarProducer::OscarProducer(edm::ParameterSet const & p)
{   
    StaticRandomEngineSetUnset random;
    m_engine = random.getEngine();
    
    produces<edm::SimTrackContainer>().setBranchAlias("SimTracks");
    produces<edm::SimVertexContainer>().setBranchAlias("SimVertices");
    produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIBLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIBHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIDLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIDHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTOBLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTOBHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTECLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTECHighTof");
    
    produces<edm::PSimHitContainer>("TotemHitsT1");
    produces<edm::PSimHitContainer>("TotemHitsT2Gem");
    produces<edm::PSimHitContainer>("TotemHitsRP");
    produces<edm::PSimHitContainer>("FP420SI");
    produces<edm::PSimHitContainer>("BSCHits");
    produces<edm::PSimHitContainer>("PLTHits");
    produces<edm::PSimHitContainer>("BHMHits");

    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
    produces<edm::PCaloHitContainer>("EcalHitsES");
    produces<edm::PCaloHitContainer>("HcalHits");
    produces<edm::PCaloHitContainer>("HGCHitsEE");
    produces<edm::PCaloHitContainer>("HGCHitsHEfront");
    produces<edm::PCaloHitContainer>("HGCHitsHEback");
    produces<edm::PCaloHitContainer>("CFCHits");
    produces<edm::PCaloHitContainer>("EcalHitsEK");
    produces<edm::PCaloHitContainer>("CaloHitsTk");
    produces<edm::PSimHitContainer>("MuonDTHits");
    produces<edm::PSimHitContainer>("MuonCSCHits");
    produces<edm::PSimHitContainer>("MuonRPCHits");
    produces<edm::PSimHitContainer>("MuonGEMHits");
    produces<edm::PSimHitContainer>("MuonME0Hits");
    produces<edm::PCaloHitContainer>("CastorPL");
    produces<edm::PCaloHitContainer>("CastorFI");
    produces<edm::PCaloHitContainer>("CastorBU");
    produces<edm::PCaloHitContainer>("CastorTU");
    produces<edm::PCaloHitContainer>("EcalTBH4BeamHits");
    produces<edm::PCaloHitContainer>("HcalTB06BeamHits");
    produces<edm::PCaloHitContainer>("ZDCHITS"); 
    produces<edm::PCaloHitContainer>("ChamberHits"); 
    produces<edm::PCaloHitContainer>("FibreHits"); 
    produces<edm::PCaloHitContainer>("WedgeHits"); 
    
    //m_runManager = RunManager::init(p);
    m_runManager = new RunManager(p);

    //register any products 
    m_producers= m_runManager->producers();

    for(Producers::iterator itProd = m_producers.begin();
	itProd != m_producers.end();
	++itProd) {
       (*itProd)->registerProducts(*this);
    }

    //UIsession manager for message handling
    m_UIsession = new CustomUIsession();

}

OscarProducer::~OscarProducer() 
{ 
  //this is causing a seg fault when an exception occurs while constructing
  // an HcalSD.  Need to check for memory problems. 
  if (m_runManager!=0) delete m_runManager; 
  if (m_UIsession!=0) delete m_UIsession;

}

void OscarProducer::beginRun(const edm::Run & r, const edm::EventSetup & es)
{
  m_runManager->initG4(es);
}


void OscarProducer::beginJob()
{
  StaticRandomEngineSetUnset random(m_engine);
}
 
void OscarProducer::endJob() { }
 
void OscarProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
    StaticRandomEngineSetUnset random(m_engine);

    std::vector<SensitiveTkDetector*>& sTk = m_runManager->sensTkDetectors();
    std::vector<SensitiveCaloDetector*>& sCalo = m_runManager->sensCaloDetectors();

    try
    {
    m_runManager->produce(e,es);

    std::auto_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
    std::auto_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
    G4SimEvent * evt = m_runManager->simEvent();
    evt->load(*p1);
    evt->load(*p2);   

    e.put(p1);
    e.put(p2);

    for (std::vector<SensitiveTkDetector*>::iterator it = sTk.begin(); it != sTk.end(); it++)
    {
	std::vector<std::string> v = (*it)->getNames();
	for (std::vector<std::string>::iterator in = v.begin(); in!= v.end(); in++)
	{
	    std::auto_ptr<edm::PSimHitContainer> product(new edm::PSimHitContainer);
 	    (*it)->fillHits(*product,*in);
	    e.put(product,*in);
	}
    }
    for (std::vector<SensitiveCaloDetector*>::iterator it = sCalo.begin(); it != sCalo.end(); it++)
    {
	std::vector<std::string>  v = (*it)->getNames();
	for (std::vector<std::string>::iterator in = v.begin(); in!= v.end(); in++)
	{
	    std::auto_ptr<edm::PCaloHitContainer> product(new edm::PCaloHitContainer);
	    (*it)->fillHits(*product,*in);
	    e.put(product,*in);
	}
    }

    for(Producers::iterator itProd = m_producers.begin();
	itProd != m_producers.end();
	++itProd) {
       (*itProd)->produce(e,es);
    }
    }
    catch ( const SimG4Exception& simg4ex )
    {
       
       edm::LogInfo("SimG4CoreApplication") << " SimG4Exception caght !" << simg4ex.what() << std::endl ;
       
       m_runManager->abortEvent() ;
       throw edm::Exception( edm::errors::EventCorruption ) ;
    }
}


StaticRandomEngineSetUnset::StaticRandomEngineSetUnset() {

    using namespace edm;
    Service<RandomNumberGenerator> rng;

    if ( ! rng.isAvailable()) {
        throw cms::Exception("Configuration")
            << "The OscarProducer module requires the RandomNumberGeneratorService\n"
               "which is not present in the configuration file.  You must add the service\n"
               "in the configuration file if you want to run OscarProducer";
    }
    m_currentEngine = &(rng->getEngine());

    m_previousEngine = CLHEP::HepRandom::getTheEngine();
    CLHEP::HepRandom::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(CLHEP::HepRandomEngine * engine) {

    m_currentEngine = engine;

    m_previousEngine = CLHEP::HepRandom::getTheEngine();
    CLHEP::HepRandom::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::~StaticRandomEngineSetUnset() {
    CLHEP::HepRandom::setTheEngine(m_previousEngine);
}

CLHEP::HepRandomEngine*
StaticRandomEngineSetUnset::getEngine() const { return m_currentEngine; }

DEFINE_FWK_MODULE(OscarProducer);
 
