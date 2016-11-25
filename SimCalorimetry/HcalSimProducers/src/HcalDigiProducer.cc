#include "SimCalorimetry/HcalSimProducers/interface/HcalDigiProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const& pset, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod(),
  theDigitizer_(pset, iC) {
  mixMod.produces<HBHEDigiCollection>();
  mixMod.produces<HODigiCollection>();
  mixMod.produces<HFDigiCollection>();
  mixMod.produces<ZDCDigiCollection>();
  mixMod.produces<QIE10DigiCollection>("HFQIE10DigiCollection");
  mixMod.produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");

}


HcalDigiProducer::HcalDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod(),
  theDigitizer_(pset, iC) {
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

void
HcalDigiProducer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setHBHENoiseSignalGenerator(noiseGenerator);
}

void
HcalDigiProducer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setHFNoiseSignalGenerator(noiseGenerator);
}

void
HcalDigiProducer::setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setHONoiseSignalGenerator(noiseGenerator);
}

void
HcalDigiProducer::setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setZDCNoiseSignalGenerator(noiseGenerator);
}

void
HcalDigiProducer::setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setQIE10NoiseSignalGenerator(noiseGenerator);
}

void
HcalDigiProducer::setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  theDigitizer_.setQIE11NoiseSignalGenerator(noiseGenerator);
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
