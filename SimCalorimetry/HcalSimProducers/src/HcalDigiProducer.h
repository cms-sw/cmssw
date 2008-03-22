#ifndef HcalDigiProducer_h
#define HcalDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"



class HcalDigiProducer : public edm::EDProducer
{
public:

  explicit HcalDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalDigiProducer();

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

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer* theHODigitizer;
  HFDigitizer* theHFDigitizer;
  ZDCDigitizer* theZDCDigitizer;

  bool doZDC;

};

#endif

