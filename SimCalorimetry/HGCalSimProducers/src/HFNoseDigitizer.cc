#include "SimCalorimetry/HGCalSimProducers/interface/HFNoseDigitizer.h"

using namespace hgc_digi;

//
HFNoseDigitizer::HFNoseDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) { }

//
void HFNoseDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,
				  const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
				  uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HFNoseDigitizer::~HFNoseDigitizer() { }

