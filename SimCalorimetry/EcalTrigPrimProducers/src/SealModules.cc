
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimProducer.h"
// wait for Paolo #include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimProducer)

  // wait for Paolo DEFINE_ANOTHER_FWK_MODULE(EcalTrigPrimAnalyzer)

