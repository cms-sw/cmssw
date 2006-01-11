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
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"


using namespace cms;


class HcalDigiProducer : public edm::EDProducer
{
public:

  explicit HcalDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer & hits);
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits> HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits> HFDigitizer;

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer* theHODigitizer;
  HFDigitizer* theHFDigitizer;

  CaloVSimParameterMap * theParameterMap;
  CaloVShape * theHcalShape;
  CaloVShape * theHFShape;
  CaloVShape * theHcalIntegratedShape;
  CaloVShape * theHFIntegratedShape;



  CaloHitResponse * theHcalResponse;
  CaloHitResponse * theHFResponse;

  HcalNoisifier * theNoisifier;
  HcalElectronicsSim * theElectronicsSim;

  std::vector<PCaloHit> theHBHEHits, theHOHits, theHFHits;
};

#endif

