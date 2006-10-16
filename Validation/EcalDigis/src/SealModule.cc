#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/EcalDigis/interface/EcalDigisValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalDigisValidation)

#include <Validation/EcalDigis/interface/EcalBarrelDigisValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalBarrelDigisValidation)

#include <Validation/EcalDigis/interface/EcalEndcapDigisValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalEndcapDigisValidation)

#include <Validation/EcalDigis/interface/EcalPreshowerDigisValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalPreshowerDigisValidation)

#include <Validation/EcalDigis/interface/EcalMixingModuleValidation.h>
DEFINE_ANOTHER_FWK_MODULE(EcalMixingModuleValidation)
