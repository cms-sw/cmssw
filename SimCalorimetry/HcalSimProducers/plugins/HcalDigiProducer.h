#ifndef SimCalorimetry_HcalSimProducers_HcalDigiProducer_h
#define SimCalorimetry_HcalSimProducers_HcalDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"

namespace edm {
  class ConsumesCollector;
  namespace one {
    class EDProducerBase;
  }
  class ParameterSet;
}

class HcalDigiProducer : public DigiAccumulatorMixMod {
public:
  HcalDigiProducer(edm::ParameterSet const& pset, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
  virtual void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  virtual void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  virtual void accumulate(edm::Event const&, edm::EventSetup const&) override;
  virtual void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
private:
  HcalDigitizer theDigitizer_;
};

#endif
