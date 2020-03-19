#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigiProducer.h"

HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const &pset,
                                   edm::ProducesCollector producesCollector,
                                   edm::ConsumesCollector &iC)
    : DigiAccumulatorMixMod(), theDigitizer_(pset, iC) {
  producesCollector.produces<HBHEDigiCollection>();
  producesCollector.produces<HODigiCollection>();
  producesCollector.produces<HFDigiCollection>();
  producesCollector.produces<ZDCDigiCollection>();
  producesCollector.produces<QIE10DigiCollection>("HFQIE10DigiCollection");
  producesCollector.produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");
  if (pset.getParameter<bool>("debugCaloSamples")) {
    producesCollector.produces<CaloSamplesCollection>("HcalSamples");
  }
  if (pset.getParameter<bool>("injectTestHits")) {
    producesCollector.produces<edm::PCaloHitContainer>("HcalHits");
  }
}

HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const &pset, edm::ConsumesCollector &iC)
    : DigiAccumulatorMixMod(), theDigitizer_(pset, iC) {}

void HcalDigiProducer::initializeEvent(edm::Event const &event, edm::EventSetup const &es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());
  theDigitizer_.initializeEvent(event, es);
}

void HcalDigiProducer::finalizeEvent(edm::Event &event, edm::EventSetup const &es) {
  theDigitizer_.finalizeEvent(event, es, randomEngine_);
  randomEngine_ = nullptr;  // to prevent access outside event
}

void HcalDigiProducer::accumulate(edm::Event const &event, edm::EventSetup const &es) {
  theDigitizer_.accumulate(event, es, randomEngine_);
}

void HcalDigiProducer::accumulate(PileUpEventPrincipal const &event,
                                  edm::EventSetup const &es,
                                  edm::StreamID const &streamID) {
  theDigitizer_.accumulate(event, es, randomEngine_);
}

void HcalDigiProducer::beginRun(edm::Run const &, edm::EventSetup const &es) {}

void HcalDigiProducer::endRun(edm::Run const &, edm::EventSetup const &) {}

void HcalDigiProducer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setHBHENoiseSignalGenerator(noiseGenerator);
}

void HcalDigiProducer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setHFNoiseSignalGenerator(noiseGenerator);
}

void HcalDigiProducer::setHONoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setHONoiseSignalGenerator(noiseGenerator);
}

void HcalDigiProducer::setZDCNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setZDCNoiseSignalGenerator(noiseGenerator);
}

void HcalDigiProducer::setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setQIE10NoiseSignalGenerator(noiseGenerator);
}

void HcalDigiProducer::setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  theDigitizer_.setQIE11NoiseSignalGenerator(noiseGenerator);
}
