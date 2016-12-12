#ifndef HcalSimProducers_HcalDigitizer_h
#define HcalSimProducers_HcalDigitizer_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalQIE1011Traits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalHitRelabeller.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/HcalCalibObjects/interface/HEDarkening.h"
#include "DataFormats/HcalCalibObjects/interface/HFRecalibration.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <vector>

class CaloHitResponse;
class HcalSimParameterMap;
class HcalAmplifier;
class HPDIonFeedbackSim;
class HcalCoderFactory;
class HcalElectronicsSim;
class HcalTimeSlewSim;
class HcalBaseSignalGenerator;
class HcalShapes;
class PileUpEventPrincipal;
class HcalTopology;

namespace edm {
  class ConsumesCollector;
}

namespace CLHEP {
  class HepRandomEngine;
}

class HcalDigitizer
{
public:

  explicit HcalDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  virtual ~HcalDigitizer();

  /**Produces the EDM products,*/
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);
  void beginRun(const edm::EventSetup & es);
  void endRun();
  
  void setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);

private:
  void accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHits, edm::Handle<std::vector<PCaloHit> > const& zdcHits, int bunchCrossing, CLHEP::HepRandomEngine*, const HcalTopology *h);

  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);
  const CaloGeometry * theGeometry;
  const HcalDDDRecConstants * theRecNumber;
  void updateGeometry(const edm::EventSetup& eventSetup);

  void buildHOSiPMCells(const std::vector<DetId>& allCells, const edm::EventSetup& eventSetup);
  void buildHFQIECells(const std::vector<DetId>& allCells, const edm::EventSetup& eventSetup);
  void buildHBHEQIECells(const std::vector<DetId>& allCells, const edm::EventSetup& eventSetup);

  //function to evaluate aging at the digi level
  void darkening(std::vector<PCaloHit>& hcalHits);
  
  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits,CaloTDigitizerQIE8Run> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits,CaloTDigitizerQIE8Run>   HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits,CaloTDigitizerQIE8Run>   HFDigitizer;
  typedef CaloTDigitizer<ZDCDigitizerTraits,CaloTDigitizerQIE8Run>  ZDCDigitizer;
  typedef CaloTDigitizer<HcalQIE10DigitizerTraits,CaloTDigitizerQIE1011Run> QIE10Digitizer;
  typedef CaloTDigitizer<HcalQIE11DigitizerTraits,CaloTDigitizerQIE1011Run> QIE11Digitizer;

  HcalSimParameterMap * theParameterMap;
  HcalShapes * theShapes;

  CaloHitResponse * theHBHEResponse;
  CaloHitResponse * theHBHESiPMResponse;
  CaloHitResponse * theHOResponse;
  CaloHitResponse * theHOSiPMResponse;
  CaloHitResponse * theHFResponse;
  CaloHitResponse * theHFQIE10Response;
  CaloHitResponse * theZDCResponse;

  // we need separate amplifiers (and electronicssims)
  // because they might have separate noise generators
  HcalAmplifier * theHBHEAmplifier;
  HcalAmplifier * theHFAmplifier;
  HcalAmplifier * theHOAmplifier;
  HcalAmplifier * theZDCAmplifier;
  HcalAmplifier * theHFQIE10Amplifier;
  HcalAmplifier * theHBHEQIE11Amplifier;

  HPDIonFeedbackSim * theIonFeedback;
  HcalCoderFactory * theCoderFactory;

  HcalElectronicsSim * theHBHEElectronicsSim;
  HcalElectronicsSim * theHFElectronicsSim;
  HcalElectronicsSim * theHOElectronicsSim;
  HcalElectronicsSim * theZDCElectronicsSim;
  HcalElectronicsSim * theHFQIE10ElectronicsSim;
  HcalElectronicsSim * theHBHEQIE11ElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HBHEHitFilter theHBHEQIE11HitFilter;
  HFHitFilter   theHFHitFilter;
  HFHitFilter   theHFQIE10HitFilter;
  HOHitFilter theHOHitFilter;
  HOHitFilter theHOSiPMHitFilter;
  ZDCHitFilter  theZDCHitFilter;

  HcalTimeSlewSim * theTimeSlewSim;

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer* theHODigitizer;
  HODigitizer* theHOSiPMDigitizer;
  HFDigitizer* theHFDigitizer;
  ZDCDigitizer* theZDCDigitizer;
  QIE10Digitizer * theHFQIE10Digitizer;
  QIE11Digitizer * theHBHEQIE11Digitizer;
  HcalHitRelabeller* theRelabeller;

  // need to cache some DetIds for the digitizers,
  // if they don't come straight from the geometry
  std::vector<DetId> hbheCells;
  std::vector<DetId> theHBHEQIE8DetIds, theHBHEQIE11DetIds;
  std::vector<DetId> theHOHPDDetIds;
  std::vector<DetId> theHOSiPMDetIds;
  std::vector<DetId> theHFQIE8DetIds, theHFQIE10DetIds;

  bool isZDC,isHCAL,zdcgeo,hbhegeo,hogeo,hfgeo;
  bool testNumbering_;
  bool doHFWindow_;
  bool killHE_;
  bool debugCS_;
  bool ignoreTime_;
  bool injectTestHits_;

  std::string hitsProducer_;

  int theHOSiPMCode;
  
  double deliveredLumi;
  HEDarkening* m_HEDarkening;
  HFRecalibration* m_HFRecalibration;

  std::vector<double> injectedHitsEnergy_;
  std::vector<double> injectedHitsTime_;
  std::vector<int> injectedHitsCells_;
  std::vector<PCaloHit> injectedHits_;
};

#endif


 
