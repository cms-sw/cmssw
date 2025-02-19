#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTrigPrimDigiProducer.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTTPDigiProducer.h"
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTTPTriggerRecord.h"


DEFINE_FWK_MODULE(HcalTrigPrimDigiProducer);
DEFINE_FWK_MODULE(HcalTTPDigiProducer);
DEFINE_FWK_MODULE(HcalTTPTriggerRecord);
