#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/HGCSimProducers/plugins/HGCDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"

//
HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::EDProducer& mixMod) :
  DigiAccumulatorMixMod(),
  theDigitizer_(new HGCDigitizer(pset) ) 
{
  if( theDigitizer_->producesEEDigis()     )
    mixMod.produces<HGCEEDigiCollection>(theDigitizer_->digiCollection());
  if( theDigitizer_->producesHEfrontDigis() || theDigitizer_->producesHEbackDigis() )
    mixMod.produces<HGCHEDigiCollection>(theDigitizer_->digiCollection());
}

//
HGCDigiProducer::~HGCDigiProducer()
{
}

//
void HGCDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) 
{
  theDigitizer_->initializeEvent(event, es);
}

//
void HGCDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) 
{
  theDigitizer_->finalizeEvent(event, es);
}

//
void HGCDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) 
{
  theDigitizer_->accumulate(event, es);
}

void HGCDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es) 
{
  theDigitizer_->accumulate(event, es);
}

//
void HGCDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) 
{
  theDigitizer_->beginRun(es);
}

//
void HGCDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) 
{
  theDigitizer_->endRun();
}

DEFINE_DIGI_ACCUMULATOR(HGCDigiProducer);
