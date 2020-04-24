#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEfrontDigitizer.h"

using namespace hgc_digi;

//
HGCHEfrontDigitizer::HGCHEfrontDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps) {
}

//
void HGCHEfrontDigitizer::runDigitizer(std::unique_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
				       const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
				       uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HGCHEfrontDigitizer::~HGCHEfrontDigitizer() { }

