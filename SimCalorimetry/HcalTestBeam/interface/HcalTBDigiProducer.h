#ifndef SimCalorimetry_HcalTestBeam_HcalTBDigiProducer_h
#define SimCalorimetry_HcalTestBeam_HcalTBDigiProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/HcalTestBeam/interface/HcalTBSimParameterMap.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"

#include <string>
#include <vector>

class PEcalTBInfo;

namespace edm {
  class StreamID;
  class ConsumesCollector;
}  // namespace edm

namespace CLHEP {
  class HepRandomEngine;
}

class HcalTBDigiProducer : public DigiAccumulatorMixMod {
public:
  explicit HcalTBDigiProducer(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &iC);
  ~HcalTBDigiProducer() override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(PileUpEventPrincipal const &e, edm::EventSetup const &c, edm::StreamID const &) override;
  void finalizeEvent(edm::Event &e, edm::EventSetup const &c) override;

private:
  void accumulateCaloHits(edm::Handle<std::vector<PCaloHit>> const &hits, int bunchCrossing);

  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer &hits);
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup &eventSetup);
  void updateGeometry();

  void setPhaseShift(const DetId &detId);

  const HcalTimeSlew *hcalTimeSlew_delay_;

private:
  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits> HODigitizer;

  HcalTBSimParameterMap *theParameterMap;
  HcalSimParameterMap *paraMap;
  CaloVShape *theHcalShape;
  CaloVShape *theHcalIntegratedShape;

  CaloHitResponse *theHBHEResponse;
  CaloHitResponse *theHOResponse;

  HcalAmplifier *theAmplifier;
  HcalCoderFactory *theCoderFactory;
  HcalElectronicsSim *theElectronicsSim;

  HBHEHitFilter theHBHEHitFilter;
  HOHitFilter theHOHitFilter;

  HcalTimeSlewSim *theTimeSlewSim;

  HBHEDigitizer *theHBHEDigitizer;
  HODigitizer *theHODigitizer;

  const CaloGeometry *theGeometry;
  std::vector<DetId> hbheCells;
  std::vector<DetId> hoCells;

  std::vector<PCaloHit> theHBHEHits, theHOHits;

  std::string ecalTBInfoLabel;
  double thisPhaseShift;

  bool doPhaseShift;
  double tunePhaseShift;

  CLHEP::HepRandomEngine *randomEngine_ = nullptr;
};

#endif
