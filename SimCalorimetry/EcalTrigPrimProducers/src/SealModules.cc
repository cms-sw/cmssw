
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimProducer.h"
#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimAnalyzer.h"
#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTPInputAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimProducer);
DEFINE_ANOTHER_FWK_MODULE(EcalTPInputAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimAnalyzer);

