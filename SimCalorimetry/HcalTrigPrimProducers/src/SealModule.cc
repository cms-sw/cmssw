#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTrigPrimDigiProducer.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalDummyHitProducer.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/TPGntupler.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalTrigPrimDigiProducer);
DEFINE_ANOTHER_FWK_MODULE(HcalDummyHitProducer);
DEFINE_ANOTHER_FWK_MODULE(TPGntupler);
