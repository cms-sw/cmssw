#include "SimMuon/Neutron/src/EDMNeutronWriter.h"
#include "FWCore/Framework/interface/Event.h"

EDMNeutronWriter::EDMNeutronWriter()
: theEvent(0),
  theHits(0)
{
}


EDMNeutronWriter::~EDMNeutronWriter() {}


void EDMNeutronWriter::writeCluster(int detType, const edm::PSimHitContainer & simHits)
{
  theHits->insert(theHits->end(), simHits.begin(), simHits.end());
}


void EDMNeutronWriter::beginEvent(edm::Event & e, const edm::EventSetup & es)
{
  theEvent = &e;
  theHits = std::auto_ptr<edm::PSimHitContainer>(new edm::PSimHitContainer());
}

void EDMNeutronWriter::endEvent()
{
  theEvent->put(theHits);
}


