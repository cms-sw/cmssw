#ifndef HcalSimProducers_HcalDigitizer_h
#define HcalSimProducers_HcalDigitizer_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class CaloVShape;
class CaloHitResponse;
class HcalSimParameterMap;
class HcalAmplifier;
class HcalCoderFactory;
class HcalElectronicsSim;
class HcalHitCorrection;
class CaloVNoiseSignalGenerator;

class HcalDigitizer
{
public:

  explicit HcalDigitizer(const edm::ParameterSet& ps);
  virtual ~HcalDigitizer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits> HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits> HFDigitizer;
  typedef CaloTDigitizer<ZDCDigitizerTraits> ZDCDigitizer;
 
  HcalSimParameterMap * theParameterMap;
  CaloVShape * theHcalShape;
  CaloVShape * theHFShape;
  CaloVShape * theZDCShape;
  CaloVShape * theHcalIntegratedShape;
  CaloVShape * theHFIntegratedShape;
  CaloVShape * theZDCIntegratedShape;

  CaloHitResponse * theHBHEResponse;
  CaloHitResponse * theHOResponse;
  CaloHitResponse * theHFResponse;
  CaloHitResponse * theZDCResponse;

  HcalAmplifier * theAmplifier;
  HcalCoderFactory * theCoderFactory;
  HcalElectronicsSim * theElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HFHitFilter   theHFHitFilter;
  HOHitFilter   theHOHitFilter;
  ZDCHitFilter  theZDCHitFilter;

  HcalHitCorrection * theHitCorrection;
  CaloVNoiseSignalGenerator * theHPDNoiseGenerator;

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer* theHODigitizer;
  HFDigitizer* theHFDigitizer;
  ZDCDigitizer* theZDCDigitizer;

  bool doZDC;

  std::string hitsProducer_;

};

#endif

