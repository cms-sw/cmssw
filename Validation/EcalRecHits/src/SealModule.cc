#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/EcalRecHits/interface/EcalRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalBarrelRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalEndcapRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalPreshowerRecHitsValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalRecHitsValidation);
DEFINE_ANOTHER_FWK_MODULE(EcalBarrelRecHitsValidation);
DEFINE_ANOTHER_FWK_MODULE(EcalEndcapRecHitsValidation);
DEFINE_ANOTHER_FWK_MODULE(EcalPreshowerRecHitsValidation);
