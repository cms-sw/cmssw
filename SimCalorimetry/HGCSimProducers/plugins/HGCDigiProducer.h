#ifndef SimCalorimetry_HGCSimProducers_HGCDigiProducer_h
#define SimCalorimetry_HGCSimProducers_HGCDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizer.h"

#include <memory>

namespace edm {
  class EDProducer;
  class ParameterSet;
}

class HGCDigiProducer : public DigiAccumulatorMixMod {
public:
  HGCDigiProducer(edm::ParameterSet const& pset, edm::EDProducer& mixMod);
  virtual void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  virtual void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  virtual void accumulate(edm::Event const&, edm::EventSetup const&) override;
  virtual void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  ~HGCDigiProducer();
private:
  //the digitizer
  std::unique_ptr<HGCDigitizer> theDigitizer_;
};

#endif
