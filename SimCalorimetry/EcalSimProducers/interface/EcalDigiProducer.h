#ifndef SimCalorimetry_EcalSimProducers_EcalDigiProducer_h
#define SimCalorimetry_EcalSimProducers_EcalDigiProducer_h

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "DataFormats/Math/interface/Error.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShapeCollection.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include <vector>

typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer;
typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer;
typedef CaloTDigitizer<ESOldDigitizerTraits> ESOldDigitizer;

class ESDigitizer;

class APDSimParameters;
class ComponentSimParameterMap;
class EEHitResponse;
class ESHitResponse;
class CaloHitResponse;
class EcalSimParameterMap;
class EcalCoder;
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
  ComponentShapeCollection m_ComponentShapes;
  EBShape m_EBShape;
  EEShape m_EEShape;
  ESShape m_ESShape;  // no const because gain must be set

  const std::string m_EBdigiCollection;
  const std::string m_EEdigiCollection;
  const std::string m_ESdigiCollection;
  const std::string m_hitsProducerTag;

  const edm::EDGetTokenT<std::vector<PCaloHit>> m_HitsEBToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> m_HitsEEToken_;
  const edm::EDGetTokenT<std::vector<PCaloHit>> m_HitsESToken_;

  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> m_pedestalsToken;
  const edm::ESGetToken<EcalIntercalibConstantsMC, EcalIntercalibConstantsMCRcd> m_icalToken;
  const edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> m_laserToken;
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> m_agcToken;
  const edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> m_grToken;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> m_geometryToken;
  edm::ESGetToken<ESGain, ESGainRcd> m_esGainToken;
  edm::ESGetToken<ESMIPToGeVConstant, ESMIPToGeVConstantRcd> m_esMIPToGeVToken;
  edm::ESGetToken<ESPedestals, ESPedestalsRcd> m_esPedestalsToken;
  edm::ESGetToken<ESIntercalibConstants, ESIntercalibConstantsRcd> m_esMIPsToken;
  edm::ESWatcher<CaloGeometryRecord> m_geometryWatcher;

  bool m_useLCcorrection;

  const bool m_apdSeparateDigi;
  const bool m_componentSeparateDigi;

  const double m_EBs25notCont;
  const double m_EEs25notCont;

  const unsigned int m_readoutFrameSize;

protected:
  std::unique_ptr<const EcalSimParameterMap> m_ParameterMap;

private:
  const std::string m_apdDigiTag;
  std::unique_ptr<const APDSimParameters> m_apdParameters;

  const std::string m_componentDigiTag;
  std::unique_ptr<const ComponentSimParameterMap> m_componentParameters;

  std::unique_ptr<EBHitResponse> m_APDResponse;

  std::unique_ptr<EBHitResponse> m_ComponentResponse;

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
  std::unique_ptr<EBDigitizer> m_ComponentDigitizer;
  std::unique_ptr<EBDigitizer> m_BarrelDigitizer;
  std::unique_ptr<EEDigitizer> m_EndcapDigitizer;

  typedef CaloTSamples<float, 10> EcalSamples;

  typedef EcalElectronicsSim<EcalCoder, EcalSamples, EcalDataFrame> EcalElectronicsSim_Ph1;
  std::unique_ptr<EcalElectronicsSim_Ph1> m_ElectronicsSim;
  std::unique_ptr<EcalCoder> m_Coder;

  std::unique_ptr<EcalElectronicsSim_Ph1> m_APDElectronicsSim;
  std::unique_ptr<EcalCoder> m_APDCoder;
  std::unique_ptr<EcalElectronicsSim_Ph1> m_ComponentElectronicsSim;
  std::unique_ptr<EcalCoder> m_ComponentCoder;

  const CaloGeometry *m_Geometry;

  std::array<std::unique_ptr<CorrelatedNoisifier<EcalCorrMatrix>>, 3> m_EBCorrNoise;
  std::array<std::unique_ptr<CorrelatedNoisifier<EcalCorrMatrix>>, 3> m_EECorrNoise;

  CLHEP::HepRandomEngine *randomEngine_ = nullptr;
};

#endif
