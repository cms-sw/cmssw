#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarProducer.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"

#include <iostream>

OscarProducer::OscarProducer(edm::ParameterSet const & p) 
{    
    produces<edm::EmbdSimTrackContainer>();
    produces<edm::EmbdSimVertexContainer>();
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
    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
    produces<edm::PCaloHitContainer>("EcalHitsES");
    produces<edm::PCaloHitContainer>("HcalHits");
    produces<edm::PCaloHitContainer>("CaloHitsTk");
    produces<edm::PSimHitContainer>("MuonDTHits");
    produces<edm::PSimHitContainer>("MuonCSCHits");
    produces<edm::PSimHitContainer>("MuonRPCHits");
    m_runManager = RunManager::init(p);

    //register any products 
    m_producers= m_runManager->producers();

    for(Producers::iterator itProd = m_producers.begin();
	itProd != m_producers.end();
	++itProd) {
       (*itProd)->registerProducts(*this);
    }
}

OscarProducer::~OscarProducer() 
{ 
  //this is causing a seg fault when an exception occurs while constructing
  // an HcalSD.  Need to check for memory problems. 
  if (m_runManager!=0) delete m_runManager; 
}

void OscarProducer::beginJob(const edm::EventSetup & es)
{
    std::cout << " OscarProducer initializing " << std::endl;
    m_runManager->initG4(es);
}
 
void OscarProducer::endJob()
{ std::cout << " OscarProducer terminating " << std::endl; }
 
void OscarProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
    std::vector<SensitiveTkDetector*>& sTk = m_runManager->sensTkDetectors();
    std::vector<SensitiveCaloDetector*>& sCalo = m_runManager->sensCaloDetectors();

    m_runManager->produce(e,es);

    std::auto_ptr<edm::EmbdSimTrackContainer> p1(new edm::EmbdSimTrackContainer);
    std::auto_ptr<edm::EmbdSimVertexContainer> p2(new edm::EmbdSimVertexContainer);
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
 
DEFINE_FWK_MODULE(OscarProducer)
 
