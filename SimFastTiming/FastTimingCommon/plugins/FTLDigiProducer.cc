#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimFastTiming/FastTimingCommon/plugins/FTLDigiProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
FTLDigiProducer::FTLDigiProducer(edm::ParameterSet const& pset,
                                 edm::ProducesCollector producesCollector,
                                 edm::ConsumesCollector& iC)
    : DigiAccumulatorMixMod() {
  std::vector<std::string> psetNames;

  pset.getParameterSetNames(psetNames);

  for (const auto& psname : psetNames) {
    const auto& ps = pset.getParameterSet(psname);
    const std::string& digitizerName = ps.getParameter<std::string>("digitizerName");
    theDigitizers_.emplace_back(FTLDigitizerFactory::get()->create(digitizerName, ps, producesCollector, iC));
  }
}

//
FTLDigiProducer::~FTLDigiProducer() {}

//
void FTLDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());
  for (auto& digitizer : theDigitizers_) {
    digitizer->initializeEvent(event, es);
  }
}

//
void FTLDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) {
  for (auto& digitizer : theDigitizers_) {
    digitizer->finalizeEvent(event, es, randomEngine_);
  }
  randomEngine_ = nullptr;  // to prevent access outside event
}

//
void FTLDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  for (auto& digitizer : theDigitizers_) {
    digitizer->accumulate(event, es, randomEngine_);
  }
}

void FTLDigiProducer::accumulate(PileUpEventPrincipal const& event,
                                 edm::EventSetup const& es,
                                 edm::StreamID const& streamID) {
  for (auto& digitizer : theDigitizers_) {
    digitizer->accumulate(event, es, randomEngine_);
  }
}

//
void FTLDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) {
  for (auto& digitizer : theDigitizers_) {
    digitizer->beginRun(es);
  }
}

//
void FTLDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) {
  for (auto& digitizer : theDigitizers_) {
    digitizer->endRun();
  }
}
