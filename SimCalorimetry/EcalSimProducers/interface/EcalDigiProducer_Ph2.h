#ifndef SimCalorimetry_EcalSimProducers_EcalDigiProducer_Ph2_h
#define SimCalorimetry_EcalSimProducers_EcalDigiProducer_Ph2_h

#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "DataFormats/Math/interface/Error.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include <vector>
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class APDSimParameters;
class CaloHitResponse;
class EcalSimParameterMap;
class EcalLiteDTUCoder;

class EcalBaseSignalGenerator;
class CaloGeometry;
class EBDigiCollectionPh2;
class PileUpEventPrincipal;

namespace edm {
  class ConsumesCollector;
  class ProducerBase;
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

class EcalDigiProducer_Ph2 : public DigiAccumulatorMixMod {
public:
  typedef EcalTDigitizer<EBDigitizerTraits_Ph2> EBDigitizer_Ph2;
  typedef EBDigitizerTraits_Ph2::ElectronicsSim EcalElectronicsSim_Ph2;

  EcalDigiProducer_Ph2(const edm::ParameterSet& params,
                       edm::ProducesCollector producesCollector,
                       edm::ConsumesCollector& iC);
  EcalDigiProducer_Ph2(const edm::ParameterSet& params, edm::ConsumesCollector& iC);
  ~EcalDigiProducer_Ph2() override;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;
  void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;

  void setEBNoiseSignalGenerator(EcalBaseSignalGenerator* noiseGenerator);

private:
  virtual void cacheEBDigis(const EBDigiCollectionPh2* ebDigiPtr) const {}

  typedef edm::Handle<std::vector<PCaloHit> > HitsHandle;

  edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> pedestalToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserToken_;
  edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> agcToken_;
  edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> icalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  void accumulateCaloHits(HitsHandle const& ebHandle, int bunchCrossing);

  void checkGeometry(const edm::EventSetup& eventSetup);

  void updateGeometry();

  void checkCalibrations(const edm::Event& event, const edm::EventSetup& eventSetup);

  APDShape m_APDShape;
  EBShape m_EBShape;

  const std::string m_EBdigiCollection;
  const std::string m_hitsProducerTag;

  bool m_useLCcorrection;

  const bool m_apdSeparateDigi;

  const double m_EBs25notCont;

  const unsigned int m_readoutFrameSize;

protected:
  std::unique_ptr<const EcalSimParameterMap> m_ParameterMap;

private:
  const std::string m_apdDigiTag;
  std::unique_ptr<const APDSimParameters> m_apdParameters;

  std::unique_ptr<EBHitResponse_Ph2> m_APDResponse;

protected:
  std::unique_ptr<EBHitResponse_Ph2> m_EBResponse;

private:
  const bool m_PreMix1;
  const bool m_PreMix2;

  std::unique_ptr<EBDigitizer_Ph2> m_APDDigitizer;
  std::unique_ptr<EBDigitizer_Ph2> m_BarrelDigitizer;

  std::unique_ptr<EcalElectronicsSim_Ph2> m_ElectronicsSim;
  std::unique_ptr<EcalLiteDTUCoder> m_Coder;

  typedef CaloTSamples<float, ecalPh2::sampleSize> EcalSamples_Ph2;
  std::unique_ptr<EcalElectronicsSim<EcalLiteDTUCoder, EcalSamples_Ph2, EcalDataFrame_Ph2> > m_APDElectronicsSim;
  std::unique_ptr<EcalLiteDTUCoder> m_APDCoder;

  const CaloGeometry* m_Geometry;

  std::array<std::unique_ptr<CorrelatedNoisifier<EcalCorrMatrix_Ph2> >, 2> m_EBCorrNoise;

  CLHEP::HepRandomEngine* randomEngine_ = nullptr;
};

#endif
