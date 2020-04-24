#include "SimCalorimetry/HGCalSimProducers/interface/HGCEEDigitizer.h"

using namespace hgc_digi;

//
HGCEEDigitizer::HGCEEDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) { }

//
void HGCEEDigitizer::runDigitizer(std::unique_ptr<HGCEEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
				  const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
				  uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HGCEEDigitizer::~HGCEEDigitizer() { }

