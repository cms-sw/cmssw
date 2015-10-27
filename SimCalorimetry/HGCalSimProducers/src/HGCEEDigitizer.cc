#include "SimCalorimetry/HGCalSimProducers/interface/HGCEEDigitizer.h"

//
HGCEEDigitizer::HGCEEDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) { }

//
void HGCEEDigitizer::runDigitizer(std::auto_ptr<HGCEEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HGCEEDigitizer::~HGCEEDigitizer() { }

