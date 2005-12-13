#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalDigiProducer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalTrigPrimRecHitProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalDigiProducer)
DEFINE_ANOTHER_FWK_MODULE(HcalTrigPrimRecHitProducer)

