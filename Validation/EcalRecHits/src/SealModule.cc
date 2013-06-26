#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"



#include <Validation/EcalRecHits/interface/EcalRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalBarrelRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalEndcapRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalPreshowerRecHitsValidation.h>
#include <Validation/EcalRecHits/interface/EcalTBValidation.h>
DEFINE_FWK_MODULE(EcalRecHitsValidation);
DEFINE_FWK_MODULE(EcalBarrelRecHitsValidation);
DEFINE_FWK_MODULE(EcalEndcapRecHitsValidation);
DEFINE_FWK_MODULE(EcalPreshowerRecHitsValidation);
DEFINE_FWK_MODULE(EcalTBValidation);
