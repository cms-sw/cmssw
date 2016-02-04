#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalHitAnalyzer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiAnalyzer.h"



DEFINE_FWK_MODULE(HcalDigiProducer);
DEFINE_FWK_MODULE(HcalHitAnalyzer);
DEFINE_FWK_MODULE(HcalDigiAnalyzer);

