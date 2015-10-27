#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEfrontDigitizer.h"

//
HGCHEfrontDigitizer::HGCHEfrontDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps) {
}

//
void HGCHEfrontDigitizer::runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HGCHEfrontDigitizer::~HGCHEfrontDigitizer() { }

