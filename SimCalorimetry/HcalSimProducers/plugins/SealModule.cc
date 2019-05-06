#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

DEFINE_FWK_MODULE(HcalHitAnalyzer);
DEFINE_FWK_MODULE(HcalDigiAnalyzer);
DEFINE_DIGI_ACCUMULATOR(HcalDigiProducer);
