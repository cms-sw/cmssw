#ifndef CastorDigiProducer_h
#define CastorDigiProducer_h

#include "FWCore/Framework/interface/one/EDProducer.h"
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
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

#include <vector>

namespace edm {
  class StreamID;
}

namespace CLHEP {
  class HepRandomEngine;
}

class PCaloHit;
class PileUpEventPrincipal;

class CastorDigiProducer : public DigiAccumulatorMixMod {
public:

  explicit CastorDigiProducer(const edm::ParameterSet& ps, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
  virtual ~CastorDigiProducer();

  virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
  virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

private:
  

  void accumulateCaloHits(std::vector<PCaloHit> const&, int bunchCrossing, CLHEP::HepRandomEngine*);

  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer & hits);
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);

  edm::InputTag theHitsProducerTag;

  CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);

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

  std::vector<CLHEP::HepRandomEngine*> randomEngines_;
};

#endif

