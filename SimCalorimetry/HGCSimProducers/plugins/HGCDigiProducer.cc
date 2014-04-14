#include "SimCalorimetry/HGCSimProducers/plugins/HGCDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"

HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::EDProducer& mixMod) :
  DigiAccumulatorMixMod(),
  theDigitizer_(pset) {
  // mixMod.produces<HBHEDigiCollection>();
  // mixMod.produces<HODigiCollection>();
  // mixMod.produces<HFDigiCollection>();
  // mixMod.produces<ZDCDigiCollection>();
  //mixMod.produces<HBHEUpgradeDigiCollection>("HBHEUpgradeDigiCollection");
  //mixMod.produces<HFUpgradeDigiCollection>("HFUpgradeDigiCollection");
}

//
void HGCDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) 
{
  theDigitizer_.initializeEvent(event, es);
}

//
void HGCDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) 
{
  theDigitizer_.finalizeEvent(event, es);
}

//
void HGCDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) 
{
  theDigitizer_.accumulate(event, es);
}

void HGCDigiProducer::accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& es) 
{
  theDigitizer_.accumulate(event, es);
}

//
void HGCDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) 
{
  theDigitizer_.beginRun(es);
}

//
void HGCDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) 
{
  theDigitizer_.endRun();
}
