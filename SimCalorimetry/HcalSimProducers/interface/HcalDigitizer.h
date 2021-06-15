#ifndef HcalSimProducers_HcalDigitizer_h
#define HcalSimProducers_HcalDigitizer_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HFRecalibration.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalQIE1011Traits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <vector>

class CaloHitResponse;
class HcalSiPMHitResponse;
class HcalAmplifier;
class HPDIonFeedbackSim;
class HcalCoderFactory;
class HcalElectronicsSim;
class HcalTimeSlewSim;
class HcalBaseSignalGenerator;
class HcalShapes;
class PileUpEventPrincipal;
class HcalTopology;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalDigitizer {
public:
  explicit HcalDigitizer(const edm::ParameterSet &ps, edm::ConsumesCollector &iC);
  virtual ~HcalDigitizer();

  /**Produces the EDM products,*/
  void initializeEvent(edm::Event const &e, edm::EventSetup const &c);
  void accumulate(edm::Event const &e, edm::EventSetup const &c, CLHEP::HepRandomEngine *);
  void accumulate(PileUpEventPrincipal const &e, edm::EventSetup const &c, CLHEP::HepRandomEngine *);
  void finalizeEvent(edm::Event &e, edm::EventSetup const &c, CLHEP::HepRandomEngine *);

  void setHBHENoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);
  void setHFNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);
  void setHONoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);
  void setZDCNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);
  void setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);
  void setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator);

private:
  void setup(const edm::EventSetup &es);
  void accumulateCaloHits(edm::Handle<std::vector<PCaloHit>> const &hcalHits,
                          edm::Handle<std::vector<PCaloHit>> const &zdcHits,
                          int bunchCrossing,
                          CLHEP::HepRandomEngine *,
                          const HcalTopology *h);

  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup &eventSetup);
  const edm::ESGetToken<HcalDbService, HcalDbRecord> conditionsToken_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> m_HBDarkeningToken;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> m_HEDarkeningToken;
  const edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> hcalTimeSlew_delay_token_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> theGeometryToken;
  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> theRecNumberToken;
  const edm::ESGetToken<HcalQIETypes, HcalQIETypesRcd> qieTypesToken_;
  edm::ESGetToken<HcalMCParams, HcalMCParamsRcd> mcParamsToken_;
  edm::ESWatcher<CaloGeometryRecord> theGeometryWatcher_;
  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  const CaloGeometry *theGeometry;
  const HcalDDDRecConstants *theRecNumber;
  void updateGeometry(const edm::EventSetup &eventSetup);

  void buildHOSiPMCells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup);
  void buildHFQIECells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup);
  void buildHBHEQIECells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup);

  // function to evaluate aging at the digi level
  void darkening(std::vector<PCaloHit> &hcalHits);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits, CaloTDigitizerQIE8Run> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits, CaloTDigitizerQIE8Run> HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits, CaloTDigitizerQIE8Run> HFDigitizer;
  typedef CaloTDigitizer<ZDCDigitizerTraits, CaloTDigitizerQIE8Run> ZDCDigitizer;
  typedef CaloTDigitizer<HcalQIE10DigitizerTraits, CaloTDigitizerQIE1011Run> QIE10Digitizer;
  typedef CaloTDigitizer<HcalQIE11DigitizerTraits, CaloTDigitizerQIE1011Run> QIE11Digitizer;

  HcalSimParameterMap theParameterMap;
  HcalShapes theShapes;

  std::unique_ptr<CaloHitResponse> theHBHEResponse;
  std::unique_ptr<HcalSiPMHitResponse> theHBHESiPMResponse;
  std::unique_ptr<CaloHitResponse> theHOResponse;
  std::unique_ptr<HcalSiPMHitResponse> theHOSiPMResponse;
  std::unique_ptr<CaloHitResponse> theHFResponse;
  std::unique_ptr<CaloHitResponse> theHFQIE10Response;
  std::unique_ptr<CaloHitResponse> theZDCResponse;

  // we need separate amplifiers (and electronicssims)
  // because they might have separate noise generators
  std::unique_ptr<HcalAmplifier> theHBHEAmplifier;
  std::unique_ptr<HcalAmplifier> theHFAmplifier;
  std::unique_ptr<HcalAmplifier> theHOAmplifier;
  std::unique_ptr<HcalAmplifier> theZDCAmplifier;
  std::unique_ptr<HcalAmplifier> theHFQIE10Amplifier;
  std::unique_ptr<HcalAmplifier> theHBHEQIE11Amplifier;

  std::unique_ptr<HPDIonFeedbackSim> theIonFeedback;
  std::unique_ptr<HcalCoderFactory> theCoderFactory;

  std::unique_ptr<HcalElectronicsSim> theHBHEElectronicsSim;
  std::unique_ptr<HcalElectronicsSim> theHFElectronicsSim;
  std::unique_ptr<HcalElectronicsSim> theHOElectronicsSim;
  std::unique_ptr<HcalElectronicsSim> theZDCElectronicsSim;
  std::unique_ptr<HcalElectronicsSim> theHFQIE10ElectronicsSim;
  std::unique_ptr<HcalElectronicsSim> theHBHEQIE11ElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HBHEHitFilter theHBHEQIE11HitFilter;
  HFHitFilter theHFHitFilter;
  HFHitFilter theHFQIE10HitFilter;
  HOHitFilter theHOHitFilter;
  HOHitFilter theHOSiPMHitFilter;
  ZDCHitFilter theZDCHitFilter;

  std::unique_ptr<HcalTimeSlewSim> theTimeSlewSim;

  std::unique_ptr<HBHEDigitizer> theHBHEDigitizer;
  std::unique_ptr<HODigitizer> theHODigitizer;
  std::unique_ptr<HODigitizer> theHOSiPMDigitizer;
  std::unique_ptr<HFDigitizer> theHFDigitizer;
  std::unique_ptr<ZDCDigitizer> theZDCDigitizer;
  std::unique_ptr<QIE10Digitizer> theHFQIE10Digitizer;
  std::unique_ptr<QIE11Digitizer> theHBHEQIE11Digitizer;
  std::unique_ptr<HcalHitRelabeller> theRelabeller;

  // need to cache some DetIds for the digitizers,
  // if they don't come straight from the geometry
  std::vector<DetId> hbheCells;
  std::vector<DetId> theHBHEQIE8DetIds, theHBHEQIE11DetIds;
  std::vector<DetId> theHOHPDDetIds;
  std::vector<DetId> theHOSiPMDetIds;
  std::vector<DetId> theHFQIE8DetIds, theHFQIE10DetIds;

  bool isZDC, isHCAL, zdcgeo, hbhegeo, hogeo, hfgeo;
  bool testNumbering_;
  bool doHFWindow_;
  bool killHE_;
  bool debugCS_;
  bool ignoreTime_;
  bool injectTestHits_;

  std::string hitsProducer_;

  int theHOSiPMCode;

  double deliveredLumi;
  bool agingFlagHB, agingFlagHE;
  const HBHEDarkening *m_HBDarkening;
  const HBHEDarkening *m_HEDarkening;
  std::unique_ptr<HFRecalibration> m_HFRecalibration;

  const HcalTimeSlew *hcalTimeSlew_delay_;

  std::vector<double> injectedHitsEnergy_;
  std::vector<double> injectedHitsTime_;
  std::vector<int> injectedHitsCells_;
  std::vector<PCaloHit> injectedHits_;
};

#endif
