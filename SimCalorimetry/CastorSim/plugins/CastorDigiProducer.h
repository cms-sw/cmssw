#ifndef CastorDigiProducer_h
#define CastorDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/CastorSim/src/CastorDigitizerTraits.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "SimCalorimetry/CastorSim/src/CastorShape.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/src/CastorHitCorrection.h"


class CastorDigiProducer : public edm::EDProducer
{
public:

  explicit CastorDigiProducer(const edm::ParameterSet& ps);
  virtual ~CastorDigiProducer();

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
  typedef CaloTDigitizer<CastorDigitizerTraits> CastorDigitizer;
 
  CastorSimParameterMap * theParameterMap;
  CaloVShape * theCastorShape;
  CaloVShape * theCastorIntegratedShape;

  CaloHitResponse * theCastorResponse;

  CastorAmplifier * theAmplifier;
  CastorCoderFactory * theCoderFactory;
  CastorElectronicsSim * theElectronicsSim;

  CastorHitFilter theCastorHitFilter;

  CastorHitCorrection * theHitCorrection;

  CastorDigitizer* theCastorDigitizer;

  std::vector<PCaloHit> theCastorHits;

};

#endif

