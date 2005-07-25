#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarProducer.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"



#include <iostream>


OscarProducer::OscarProducer(edm::ParameterSet const & p) 
{    
    m_runManager = RunManager::init(p);
}

OscarProducer::~OscarProducer() 
{ 
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
    std::cout << " Produce " << std::endl;
    m_runManager->produce(es);

    std::auto_ptr<edm::EmbdSimTrackContainer> p1(new edm::EmbdSimTrackContainer);
    std::auto_ptr<edm::EmbdSimVertexContainer> p2(new edm::EmbdSimVertexContainer);

    G4SimEvent * evt = m_runManager->simEvent();
    evt->load(*p1);
    evt->load(*p2);

    e.put(p1);
    e.put(p2);

    std::cout << " Produced " << std::endl;
    //
    // now produce Hits
    //
    
    std::vector<SensitiveTkDetector*>& sTk = m_runManager->sensTkDetectors();
    std::vector<SensitiveCaloDetector*>& sCalo = m_runManager->sensCaloDetectors();

    //
    // Tk Hits
    //
    for (std::vector<SensitiveTkDetector*>::iterator it = sTk.begin(); it != sTk.end(); it++){
      //
      // Look whether they have to go to diff containers
      //
      std::vector<std::string>  v = (*it)->getNames();
      for (std::vector<std::string>::iterator it = v.begin(); it!= v.end(); it++){
	//
	// Create an empty container
	//
	
	
	std::auto_ptr<edm::PSimHitContainer> product(new edm::PSimHitContainer);
	//
	// fill it
	//
	(*it)->fillHits(*product,*v);
	//
	// put it with label
	//
	e.put(product,*v);
      }
    }
      
    //
    // Calo Hits
    //

    //
    // Tk Hits
    //
    for (std::vector<SensitiveCaloDetector*>::iterator it = sCalo.begin(); it != sCalo.end(); it++){
      //
      // Look whether they have to go to diff containers
      //
      std::vector<std::string>  v = (*it)->getNames();
      for (std::vector<std::string>::iterator it = v.begin(); it!= v.end(); it++){
	//
	// Create an empty container
	//
	std::auto_ptr<edm::PCaloHitContainer> product(new edm::PCaloHitContainer);
	//
	// fill it
	//
	(*it)->fillHits(*product,*v);
	//
	// put it with label
	//
	e.put(product,*v);
      }
    }

}
 
DEFINE_FWK_MODULE(OscarProducer)
 
