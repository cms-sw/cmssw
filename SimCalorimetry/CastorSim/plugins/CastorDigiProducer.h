#ifndef CastorDigiProducer_h
#define CastorDigiProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/src/CastorDigitizerTraits.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"
#include "SimCalorimetry/CastorSim/src/CastorHitCorrection.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/src/CastorShape.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

#include <vector>

namespace edm {
  class StreamID;
  class ConsumesCollector;
}  // namespace edm

namespace CLHEP {
  class HepRandomEngine;
}

class PCaloHit;
class PileUpEventPrincipal;

class CastorDigiProducer : public DigiAccumulatorMixMod {
public:
  explicit CastorDigiProducer(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &iC);
  ~CastorDigiProducer() override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(PileUpEventPrincipal const &e, edm::EventSetup const &c, edm::StreamID const &) override;
  void finalizeEvent(edm::Event &e, edm::EventSetup const &c) override;

private:
  void accumulateCaloHits(std::vector<PCaloHit> const &, int bunchCrossing);

  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer &hits);
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup &eventSetup);

  edm::InputTag theHitsProducerTag;

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<CastorDigitizerTraits> CastorDigitizer;

  CastorSimParameterMap *theParameterMap;
  CaloVShape *theCastorShape;
  CaloVShape *theCastorIntegratedShape;

  CaloHitResponse *theCastorResponse;

  CastorAmplifier *theAmplifier;
  CastorCoderFactory *theCoderFactory;
  CastorElectronicsSim *theElectronicsSim;

  CastorHitFilter theCastorHitFilter;

  CastorHitCorrection *theHitCorrection;

  CastorDigitizer *theCastorDigitizer;

  std::vector<PCaloHit> theCastorHits;

  CLHEP::HepRandomEngine *randomEngine_ = nullptr;
};

#endif
