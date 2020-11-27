#include "HFNoseDigitizer.h"

using namespace hgc_digi;

//
HFNoseDigitizer::HFNoseDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  this->det_ = DetId::Forward;
  this->subdet_ = ForwardSubdetector::HFNose;
}

//
void HFNoseDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                   HGCSimHitDataAccumulator& simData,
                                   const CaloSubdetectorGeometry* theGeom,
                                   const std::unordered_set<DetId>& validIds,
                                   CLHEP::HepRandomEngine* engine) {}

//
HFNoseDigitizer::~HFNoseDigitizer() {}
