#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalZeroSuppressionProducers/src/HcalSimpleAmplitudeZS.h"
#include "SimCalorimetry/HcalZeroSuppressionProducers/src/HcalRealisticZS.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalSimpleAmplitudeZS);
DEFINE_ANOTHER_FWK_MODULE(HcalRealisticZS);
