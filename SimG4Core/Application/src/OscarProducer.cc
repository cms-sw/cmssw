#include "PluginManager/PluginManager.h"

#include "FWCore/CoreFramework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarProducer.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

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
}
 
DEFINE_FWK_MODULE(OscarProducer)
 
