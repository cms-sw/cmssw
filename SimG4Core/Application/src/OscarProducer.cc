#include "PluginManager/PluginManager.h"

#include "FWCore/CoreFramework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarProducer.h"

OscarProducer::OscarProducer(edm::ParameterSet const & p) 
{    
    m_runManager = RunManager::init(p);
}

OscarProducer::~OscarProducer() 
{ 
    //if (m_runManager!=0) delete m_runManager; 
}

void OscarProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
    std::cout << " Produce " << std::endl;
    m_runManager->produce(es);
    std::cout << " Produced " << std::endl;
}
 
DEFINE_FWK_MODULE(OscarProducer)
 
