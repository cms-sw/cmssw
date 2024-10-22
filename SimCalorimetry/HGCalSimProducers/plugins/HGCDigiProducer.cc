#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/HGCalSimProducers/plugins/HGCDigiProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset,
                                 edm::ProducesCollector producesCollector,
                                 edm::ConsumesCollector& iC)
    : HGCDigiProducer(pset, iC) {
  premixStage1_ = pset.getParameter<bool>("premixStage1");
  if (premixStage1_) {
    producesCollector.produces<PHGCSimAccumulator>(theDigitizer_.digiCollection());
  } else {
    producesCollector.produces<HGCalDigiCollection>(theDigitizer_.digiCollection());
  }
}

HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
    : DigiAccumulatorMixMod(), theDigitizer_(pset, iC) {}

//
void HGCDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());
  theDigitizer_.initializeEvent(event, es);
}

//
void HGCDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) {
  theDigitizer_.finalizeEvent(event, es, randomEngine_);
  randomEngine_ = nullptr;  // to precent access outside event
}

//
void HGCDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  if (premixStage1_) {
    theDigitizer_.accumulate_forPreMix(event, es, randomEngine_);
  }

  else {
    theDigitizer_.accumulate(event, es, randomEngine_);
  }
}

void HGCDigiProducer::accumulate(PileUpEventPrincipal const& event,
                                 edm::EventSetup const& es,
                                 edm::StreamID const& streamID) {
  if (premixStage1_) {
    theDigitizer_.accumulate_forPreMix(event, es, randomEngine_);
  } else {
    theDigitizer_.accumulate(event, es, randomEngine_);
  }
}

DEFINE_DIGI_ACCUMULATOR(HGCDigiProducer);
