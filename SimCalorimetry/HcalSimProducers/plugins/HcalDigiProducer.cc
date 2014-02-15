#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const& pset, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod(),
  theDigitizer_(pset, iC) {
  mixMod.produces<HBHEDigiCollection>();
  mixMod.produces<HODigiCollection>();
  mixMod.produces<HFDigiCollection>();
  mixMod.produces<ZDCDigiCollection>();
  mixMod.produces<HBHEUpgradeDigiCollection>("HBHEUpgradeDigiCollection");
  mixMod.produces<HFUpgradeDigiCollection>("HFUpgradeDigiCollection");

}

void
HcalDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) {
  theDigitizer_.initializeEvent(event, es);
}

void
HcalDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) {
  theDigitizer_.finalizeEvent(event, es, randomEngine(event.streamID()));
}

void
HcalDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  theDigitizer_.accumulate(event, es, randomEngine(event.streamID()));
}

void
HcalDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es, edm::StreamID const& streamID) {
  theDigitizer_.accumulate(event, es, randomEngine(streamID));
}

void
HcalDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) {
  theDigitizer_.beginRun(es);
}

void
HcalDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) {
  theDigitizer_.endRun();
}

CLHEP::HepRandomEngine* HcalDigiProducer::randomEngine(edm::StreamID const& streamID) {
  unsigned int index = streamID.value();
  if(index >= randomEngines_.size()) {
    randomEngines_.resize(index + 1, nullptr);
  }
  CLHEP::HepRandomEngine* ptr = randomEngines_[index];
  if(!ptr) {
    edm::Service<edm::RandomNumberGenerator> rng;
    ptr = &rng->getEngine(streamID);
    randomEngines_[index] = ptr;
  }
  return ptr;
}
