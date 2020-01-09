#ifndef SimCalorimetry_EcalSimProducers_EcalDigiProducer_h
#define SimCalorimetry_EcalSimProducers_EcalDigiProducer_h

#include "DataFormats/Math/interface/Error.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

#include <vector>

typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer;
typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer;
typedef CaloTDigitizer<ESOldDigitizerTraits> ESOldDigitizer;

class ESDigitizer;

class APDSimParameters;
class EBHitResponse;
class EEHitResponse;
class ESHitResponse;
class CaloHitResponse;
class EcalSimParameterMap;
class EcalCoder;
class EcalElectronicsSim;
class ESElectronicsSim;
class ESElectronicsSimFast;
class EcalBaseSignalGenerator;
class CaloGeometry;
class EBDigiCollection;
class EEDigiCollection;
class ESDigiCollection;
class PileUpEventPrincipal;

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  template <typename T>
  class Handle;
  class ParameterSet;
  class StreamID;
}  // namespace edm

namespace CLHEP {
  class HepRandomEngine;
}

class EcalDigiProducer : public DigiAccumulatorMixMod {
public:
  EcalDigiProducer(const edm::ParameterSet &params, edm::ProducesCollector, edm::ConsumesCollector &iC);
  EcalDigiProducer(const edm::ParameterSet &params, edm::ConsumesCollector &iC);
  ~EcalDigiProducer() override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(PileUpEventPrincipal const &e, edm::EventSetup const &c, edm::StreamID const &) override;
  void finalizeEvent(edm::Event &e, edm::EventSetup const &c) override;
  void beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) override;
  void beginRun(edm::Run const &run, edm::EventSetup const &setup) override;

  void setEBNoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator);
  void setEENoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator);
  void setESNoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator);

private:
  virtual void cacheEBDigis(const EBDigiCollection *ebDigiPtr) const {}
  virtual void cacheEEDigis(const EEDigiCollection *eeDigiPtr) const {}

  typedef edm::Handle<std::vector<PCaloHit>> HitsHandle;
  void accumulateCaloHits(HitsHandle const &ebHandle,
                          HitsHandle const &eeHandle,
                          HitsHandle const &esHandle,
                          int bunchCrossing);

  void checkGeometry(const edm::EventSetup &eventSetup);

  void updateGeometry();

  void checkCalibrations(const edm::Event &event, const edm::EventSetup &eventSetup);

  APDShape m_APDShape;
  EBShape m_EBShape;
  EEShape m_EEShape;
  ESShape m_ESShape;  // no const because gain must be set

  const std::string m_EBdigiCollection;
  const std::string m_EEdigiCollection;
  const std::string m_ESdigiCollection;
  const std::string m_hitsProducerTag;

  bool m_useLCcorrection;

  const bool m_apdSeparateDigi;

  const double m_EBs25notCont;
  const double m_EEs25notCont;

  const unsigned int m_readoutFrameSize;

protected:
  std::unique_ptr<const EcalSimParameterMap> m_ParameterMap;

private:
  const std::string m_apdDigiTag;
  std::unique_ptr<const APDSimParameters> m_apdParameters;

  std::unique_ptr<EBHitResponse> m_APDResponse;

protected:
  std::unique_ptr<EBHitResponse> m_EBResponse;
  std::unique_ptr<EEHitResponse> m_EEResponse;

private:
  std::unique_ptr<ESHitResponse> m_ESResponse;
  std::unique_ptr<CaloHitResponse> m_ESOldResponse;

  const bool m_addESNoise;
  const bool m_PreMix1;
  const bool m_PreMix2;

  const bool m_doFastES;

  const bool m_doEB, m_doEE, m_doES;

  std::unique_ptr<ESElectronicsSim> m_ESElectronicsSim;
  std::unique_ptr<ESOldDigitizer> m_ESOldDigitizer;
  std::unique_ptr<ESElectronicsSimFast> m_ESElectronicsSimFast;
  std::unique_ptr<ESDigitizer> m_ESDigitizer;

  std::unique_ptr<EBDigitizer> m_APDDigitizer;
  std::unique_ptr<EBDigitizer> m_BarrelDigitizer;
  std::unique_ptr<EEDigitizer> m_EndcapDigitizer;

  std::unique_ptr<EcalElectronicsSim> m_ElectronicsSim;
  std::unique_ptr<EcalCoder> m_Coder;

  std::unique_ptr<EcalElectronicsSim> m_APDElectronicsSim;
  std::unique_ptr<EcalCoder> m_APDCoder;

  const CaloGeometry *m_Geometry;

  std::array<std::unique_ptr<CorrelatedNoisifier<EcalCorrMatrix>>, 3> m_EBCorrNoise;
  std::array<std::unique_ptr<CorrelatedNoisifier<EcalCorrMatrix>>, 3> m_EECorrNoise;

  CLHEP::HepRandomEngine *randomEngine_ = nullptr;
};

#endif
