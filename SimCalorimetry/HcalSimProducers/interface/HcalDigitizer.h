#ifndef HcalSimProducers_HcalDigitizer_h
#define HcalSimProducers_HcalDigitizer_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrices.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitRelabeller.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HcalUpgradeTraits.h"

class CaloVShape;
class CaloHitResponse;
class HcalSimParameterMap;
class HcalAmplifier;
class HPDIonFeedbackSim;
class HcalCoderFactory;
class HcalElectronicsSim;
class HcalHitCorrection;
class HcalBaseSignalGenerator;

class HcalDigitizer
{
public:

  explicit HcalDigitizer(const edm::ParameterSet& ps);
  virtual ~HcalDigitizer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  void setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);
  void setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator);

private:
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);
  const CaloGeometry * theGeometry;
  void updateGeometry();

  void buildHOSiPMCells(const std::vector<DetId>& allCells);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits> HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits> HFDigitizer;
  typedef CaloTDigitizer<ZDCDigitizerTraits> ZDCDigitizer;

  typedef CaloTDigitizer<HcalUpgradeDigitizerTraits> UpgradeDigitizer;
 
  HcalSimParameterMap * theParameterMap;
  CaloVShape * theHcalShape;
  CaloVShape * theSiPMShape;
  CaloVShape * theHFShape;
  CaloVShape * theZDCShape;
  CaloVShape * theHcalIntegratedShape;
  CaloVShape * theSiPMIntegratedShape;
  CaloVShape * theHFIntegratedShape;
  CaloVShape * theZDCIntegratedShape;

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

  HcalElectronicsSim * theUpgradeElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HFHitFilter   theHFHitFilter;
  HOHitFilter   theHOHitFilter;
  HcalHitFilter theHOSiPMHitFilter;
  ZDCHitFilter  theZDCHitFilter;

  HcalHitCorrection * theHitCorrection;
  CaloVNoiseSignalGenerator * theNoiseGenerator;
  CaloVNoiseHitGenerator * theNoiseHitGenerator;

  HBHEDigitizer * theHBHEDigitizer;
  HBHEDigitizer * theHBHESiPMDigitizer;
  HODigitizer* theHODigitizer;
  HODigitizer* theHOSiPMDigitizer;
  HFDigitizer* theHFDigitizer;
  ZDCDigitizer* theZDCDigitizer;

  UpgradeDigitizer * theUpgradeDigitizer;

  // need to cache some DetIds for the digitizers,
  // if they don't come straight from the geometry
  std::vector<DetId> theHBHEDetIds;
  std::vector<DetId> theHOHPDDetIds;
  std::vector<DetId> theHOSiPMDetIds;

  bool isZDC,isHCAL,zdcgeo,hbhegeo,hogeo,hfgeo;
  bool relabel_;
  HcalHitRelabeller* theRelabeller;

  std::string hitsProducer_;

  int theHOSiPMCode;
};

#endif


