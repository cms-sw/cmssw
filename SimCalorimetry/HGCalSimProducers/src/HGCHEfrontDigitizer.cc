#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEfrontDigitizer.h"

using namespace hgc_digi;

//
HGCHEfrontDigitizer::HGCHEfrontDigitizer(const edm::ParameterSet &ps) : HGCDigitizerBase(ps) {
}

//
void HGCHEfrontDigitizer::runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
}

//
HGCHEfrontDigitizer::~HGCHEfrontDigitizer() { }

