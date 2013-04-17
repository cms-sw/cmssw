#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"

HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const& pset, edm::EDProducer& mixMod) :
  DigiAccumulatorMixMod(),
  theDigitizer_(pset) {
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
  theDigitizer_.finalizeEvent(event, es);
}

void
HcalDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  theDigitizer_.accumulate(event, es);
}

void
HcalDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es) {
  theDigitizer_.accumulate(event, es);
}

void
HcalDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) {
  theDigitizer_.beginRun(es);
}

void
HcalDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) {
  theDigitizer_.endRun();
}
