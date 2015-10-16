#ifndef HcalSimProducers_HcalDigitizer_h
#define HcalSimProducers_HcalDigitizer_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalUpgradeTraits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalHitRelabeller.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/HcalCalibObjects/interface/HEDarkening.h"
#include "DataFormats/HcalCalibObjects/interface/HFRecalibration.h"

#include <vector>

class CaloHitResponse;
class HcalSimParameterMap;
class HcalAmplifier;
class HPDIonFeedbackSim;
class HcalCoderFactory;
class HcalElectronicsSim;
class HcalHitCorrection;
class HcalTimeSlewSim;
class HcalBaseSignalGenerator;
class HcalShapes;
class PCaloHit;
class PileUpEventPrincipal;

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

private:
  void accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHits, edm::Handle<std::vector<PCaloHit> > const& zdcHits, int bunchCrossing, CLHEP::HepRandomEngine*);

  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);
  const CaloGeometry * theGeometry;
  const HcalDDDRecConstants * theRecNumber;
  void updateGeometry(const edm::EventSetup& eventSetup);

  void buildHOSiPMCells(const std::vector<DetId>& allCells, const edm::EventSetup& eventSetup);

  //function to evaluate aging at the digi level
  void darkening(std::vector<PCaloHit>& hcalHits);
  
  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits>   HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits>   HFDigitizer;
  typedef CaloTDigitizer<ZDCDigitizerTraits>  ZDCDigitizer;
  typedef CaloTDigitizer<HcalUpgradeDigitizerTraits> UpgradeDigitizer;

  HcalSimParameterMap * theParameterMap;
  HcalShapes * theShapes;

  CaloHitResponse * theHBHEResponse;
  CaloHitResponse * theHBHESiPMResponse;
  CaloHitResponse * theHOResponse;
  CaloHitResponse * theHOSiPMResponse;
  CaloHitResponse * theHFResponse;
  CaloHitResponse * theZDCResponse;

  // we need separate amplifiers (and electronicssims)
  // because they might have separate noise generators
  HcalAmplifier * theHBHEAmplifier;
  HcalAmplifier * theHFAmplifier;
  HcalAmplifier * theHOAmplifier;
  HcalAmplifier * theZDCAmplifier;

  HPDIonFeedbackSim * theIonFeedback;
  HcalCoderFactory * theCoderFactory;
  HcalCoderFactory * theUpgradeCoderFactory;

  HcalElectronicsSim * theHBHEElectronicsSim;
  HcalElectronicsSim * theHFElectronicsSim;
  HcalElectronicsSim * theHOElectronicsSim;
  HcalElectronicsSim * theZDCElectronicsSim;
  HcalElectronicsSim * theUpgradeHBHEElectronicsSim;
  HcalElectronicsSim * theUpgradeHFElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HFHitFilter   theHFHitFilter;
  HOHitFilter   theHOHitFilter;
  HcalHitFilter theHOSiPMHitFilter;
  ZDCHitFilter  theZDCHitFilter;

  HcalHitCorrection * theHitCorrection;
  HcalTimeSlewSim * theTimeSlewSim;
  CaloVNoiseSignalGenerator * theNoiseGenerator;
  CaloVNoiseHitGenerator * theNoiseHitGenerator;

  HBHEDigitizer * theHBHEDigitizer;
  HBHEDigitizer * theHBHESiPMDigitizer;
  HODigitizer* theHODigitizer;
  HODigitizer* theHOSiPMDigitizer;
  HFDigitizer* theHFDigitizer;
  ZDCDigitizer* theZDCDigitizer;
  UpgradeDigitizer * theHBHEUpgradeDigitizer;
  UpgradeDigitizer * theHFUpgradeDigitizer;
  HcalHitRelabeller* theRelabeller;

  // need to cache some DetIds for the digitizers,
  // if they don't come straight from the geometry
  std::vector<DetId> theHBHEDetIds;
  std::vector<DetId> theHOHPDDetIds;
  std::vector<DetId> theHOSiPMDetIds;

  bool isZDC,isHCAL,zdcgeo,hbhegeo,hogeo,hfgeo;
  bool relabel_;

  std::string hitsProducer_;

  int theHOSiPMCode;
  
  double deliveredLumi;
  HEDarkening* m_HEDarkening;
  HFRecalibration* m_HFRecalibration;
};

#endif


 
