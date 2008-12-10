#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalDigiProducer);
DEFINE_ANOTHER_FWK_MODULE(HcalHitAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HcalDigiAnalyzer);

