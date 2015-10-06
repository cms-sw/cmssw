#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/HGCalSimProducers/plugins/HGCDigiProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod(),
  theDigitizer_(new HGCDigitizer(pset, iC)) {
  if (theDigitizer_->producesEEDigis())
    mixMod.produces<HGCEEDigiCollection>(theDigitizer_->digiCollection());
  if (theDigitizer_->producesHEfrontDigis() || 
      theDigitizer_->producesHEbackDigis() )
    mixMod.produces<HGCHEDigiCollection>(theDigitizer_->digiCollection());
}

HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod(),
  theDigitizer_(new HGCDigitizer(pset, iC)) {
}

HGCDigiProducer::~HGCDigiProducer() { }

void HGCDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) {
  theDigitizer_->initializeEvent(event, es);
}

void HGCDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) {
  theDigitizer_->finalizeEvent(event, es, randomEngine(event.streamID()));
}

void HGCDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  theDigitizer_->accumulate(event, es, randomEngine(event.streamID()));
}

void HGCDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es, edm::StreamID const& streamID) {
  theDigitizer_->accumulate(event, es, randomEngine(streamID));
}

void HGCDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) {
  theDigitizer_->beginRun(es);
}

void HGCDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) {
  theDigitizer_->endRun();
}

CLHEP::HepRandomEngine* HGCDigiProducer::randomEngine(edm::StreamID const& streamID) {
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

DEFINE_DIGI_ACCUMULATOR(HGCDigiProducer);
