#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimFastTiming/FastTimingCommon/plugins/MTDDigiProducer.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
MTDDigiProducer::MTDDigiProducer(edm::ParameterSet const& pset, edm::ProducerBase& mixMod,
                                 edm::ConsumesCollector& iC) :
  DigiAccumulatorMixMod() {
  std::vector<std::string> psetNames;

  pset.getParameterSetNames(psetNames);
  
  for(const auto& psname : psetNames) {
    const auto& ps = pset.getParameterSet(psname);
    const std::string& digitizerName = ps.getParameter<std::string>("digitizerName");
    auto temp = MTDDigitizerFactory::get()->create(digitizerName,ps,iC,mixMod);
    theDigitizers_.emplace_back(temp);
  } 
}

//
MTDDigiProducer::~MTDDigiProducer()
{
}

//
void MTDDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->initializeEvent(event, es);
  }
}

//
void MTDDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->finalizeEvent(event, es, randomEngine(event.streamID()));
  }
}

//
void MTDDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->accumulate(event, es, randomEngine(event.streamID()));
  }
}

void MTDDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es, edm::StreamID const& streamID) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->accumulate(event, es, randomEngine(streamID));
  }
}

//
void MTDDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->beginRun(es);
  }
}

//
void MTDDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) 
{
  for( auto& digitizer : theDigitizers_ ) {
    digitizer->endRun();
  }
}

CLHEP::HepRandomEngine* MTDDigiProducer::randomEngine(edm::StreamID const& streamID) {
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
