#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/HcalRecHits/interface/HcalRecHitsClient.h"
#include "Validation/HcalRecHits/interface/NoiseRatesClient.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalRecHitsClient);
DEFINE_ANOTHER_FWK_MODULE(NoiseRatesClient);
