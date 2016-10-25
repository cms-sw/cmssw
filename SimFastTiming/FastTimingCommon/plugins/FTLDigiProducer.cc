#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimFastTiming/FastTimingCommon/plugins/FTLDigiProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
FTLDigiProducer::FTLDigiProducer(edm::ParameterSet const& pset, edm::stream::EDProducerBase& mixMod, 
                                 edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod() {
  std::vector<std::string> psetNames;

  pset.getParameterSetNames(psetNames);
  
  for(const auto& psname : psetNames) {
    const auto& ps = pset.getParameterSet(psname);
    const std::string& digitizerName = ps.getParameter<std::string>("digitizerName");
    auto temp = FTLDigitizerFactory::get()->create(digitizerName,ps,iC,mixMod);
    theDigitizers_.emplace_back(temp);
  } 
}

//
FTLDigiProducer::~FTLDigiProducer()
{
}

//
void FTLDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->initializeEvent(event, es);
  }
}

//
void FTLDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->finalizeEvent(event, es, randomEngine(event.streamID()));
  }
}

//
void FTLDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->accumulate(event, es, randomEngine(event.streamID()));
  }
}

void FTLDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es, edm::StreamID const& streamID) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->accumulate(event, es, randomEngine(streamID));
  }
}

//
void FTLDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->beginRun(es);
  }
}

//
void FTLDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->endRun();
  }
}

CLHEP::HepRandomEngine* FTLDigiProducer::randomEngine(edm::StreamID const& streamID) {
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

DEFINE_DIGI_ACCUMULATOR(FTLDigiProducer);
