#ifndef SimCalorimetry_HGCSimProducers_HGCDigiProducer_h
#define SimCalorimetry_HGCSimProducers_HGCDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizer.h"

#include <memory>
#include <vector>

namespace edm {
  class ConsumesCollector;
  namespace one {
    class EDProducerBase;
  }
  class ParameterSet;
  class StreamID;
}

namespace CLHEP {
  class HepRandomEngine;
}

class HGCDigiProducer : public DigiAccumulatorMixMod {
public:
  HGCDigiProducer(edm::ParameterSet const& pset, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
  HGCDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC);

  virtual void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  virtual void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  virtual void accumulate(edm::Event const&, edm::EventSetup const&) override;
  virtual void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  ~HGCDigiProducer();
private:
  CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);
  //the digitizer
  std::unique_ptr<HGCDigitizer> theDigitizer_;
  std::vector<CLHEP::HepRandomEngine*> randomEngines_;
};

#endif
