#ifndef SimCalorimetry_HcalSimProducers_HcalDigiProducer_h
#define SimCalorimetry_HcalSimProducers_HcalDigiProducer_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"

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

class HcalDigiProducer : public DigiAccumulatorMixMod {
public:
  HcalDigiProducer(edm::ParameterSet const& pset, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC);

  HcalDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  void setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);

private:

  CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);

  HcalDigitizer theDigitizer_;

  std::vector<CLHEP::HepRandomEngine*> randomEngines_;
};

#endif
