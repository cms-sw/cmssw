#include "SimCalorimetry/HGCalSimProducers/interface/HGCEEDigitizer.h"

using namespace hgc_digi;

//
HGCEEDigitizer::HGCEEDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  this->det_ = DetId::HGCalEE;
}

//
void HGCEEDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                  HGCSimHitDataAccumulator& simData,
                                  const CaloSubdetectorGeometry* theGeom,
                                  const std::unordered_set<DetId>& validIds,
                                  CLHEP::HepRandomEngine* engine) {}

//
HGCEEDigitizer::~HGCEEDigitizer() {}
