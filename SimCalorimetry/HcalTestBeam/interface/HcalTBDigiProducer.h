#ifndef HcalTestBeam_HcalTBDigiProducer_h
#define HcalTestBeam_HcalTBDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalTestBeam/interface/HcalTBSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HBHEHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HOHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"

#include<vector>
#include<string>

class HcalTBDigiProducer : public edm::EDProducer {

public:

  explicit HcalTBDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalTBDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer & hits);
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);
  void updateGeometry();

  void setPhaseShift(const DetId & detId);

private:

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits>   HODigitizer;

  HcalTBSimParameterMap * theParameterMap;
  CaloVShape * theHcalShape;
  CaloVShape * theHcalIntegratedShape;

  CaloHitResponse * theHBHEResponse;
  CaloHitResponse * theHOResponse;

  HcalAmplifier * theAmplifier;
  HcalCoderFactory * theCoderFactory;
  HcalElectronicsSim * theElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HOHitFilter   theHOHitFilter;

  HcalHitCorrection * theHitCorrection;

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer   * theHODigitizer;
 
  const CaloGeometry * theGeometry;
  std::vector<DetId>   hbheCells;
  std::vector<DetId>   hoCells;

  std::vector<PCaloHit> theHBHEHits, theHOHits;

  std::string ecalTBInfoLabel;
  double thisPhaseShift;

  bool   doPhaseShift;
  double tunePhaseShift;
};

#endif

