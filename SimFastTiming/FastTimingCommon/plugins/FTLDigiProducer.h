#ifndef SimFastTiming_FastTimingCommon_FTLDigiProducer_h
#define SimFastTiming_FastTimingCommon_FTLDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerBase.h"

#include <memory>
#include <vector>

namespace edm {
  class ConsumesCollector;
  namespace stream {
    class EDProducerBase;
  }
  class ParameterSet;
  class StreamID;
}

namespace CLHEP {
  class HepRandomEngine;
}

class FTLDigiProducer : public DigiAccumulatorMixMod {
public:
  FTLDigiProducer(edm::ParameterSet const& pset, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
  FTLDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
  {
    throw cms::Exception("DeprecatedConstructor") << "Please make sure you're calling this with the threaded mixing module...";
  }

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  ~FTLDigiProducer() override;
private:
  CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);
  //the digitizer
  std::vector<std::unique_ptr<FTLDigitizerBase> > theDigitizers_;
  std::vector<CLHEP::HepRandomEngine*> randomEngines_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
DEFINE_DIGI_ACCUMULATOR(FTLDigiProducer);

#endif
