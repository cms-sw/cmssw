#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalTrigPrimDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalDigiProducer)
DEFINE_ANOTHER_FWK_MODULE(HcalTrigPrimDigiProducer)
DEFINE_ANOTHER_FWK_MODULE(HcalHitAnalyzer)
DEFINE_ANOTHER_FWK_MODULE(HcalDigiAnalyzer)

