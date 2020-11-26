#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEfrontDigitizer.h"

using namespace hgc_digi;

//
HGCHEfrontDigitizer::HGCHEfrontDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  this->det_ = DetId::HGCalHSi;
}

//
void HGCHEfrontDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                       HGCSimHitDataAccumulator& simData,
                                       const CaloSubdetectorGeometry* theGeom,
                                       const std::unordered_set<DetId>& validIds,
                                       CLHEP::HepRandomEngine* engine) {}

//
HGCHEfrontDigitizer::~HGCHEfrontDigitizer() {}
