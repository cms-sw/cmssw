#include "SimMuon/Neutron/src/EDMNeutronWriter.h"

#include <memory>

#include "FWCore/Framework/interface/Event.h"

EDMNeutronWriter::EDMNeutronWriter() : theEvent(nullptr), theHits(nullptr) {}

EDMNeutronWriter::~EDMNeutronWriter() {}

void EDMNeutronWriter::writeCluster(int detType, const edm::PSimHitContainer& simHits) {
  theHits->insert(theHits->end(), simHits.begin(), simHits.end());
}

void EDMNeutronWriter::beginEvent(edm::Event& e, const edm::EventSetup& es) {
  theEvent = &e;
  theHits = std::make_unique<edm::PSimHitContainer>();
}

void EDMNeutronWriter::endEvent() { theEvent->put(std::move(theHits)); }
