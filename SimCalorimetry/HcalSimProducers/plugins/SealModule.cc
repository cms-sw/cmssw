#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"

DEFINE_FWK_MODULE(HcalHitAnalyzer);
DEFINE_FWK_MODULE(HcalDigiAnalyzer);
DEFINE_DIGI_ACCUMULATOR(HcalDigiProducer);
