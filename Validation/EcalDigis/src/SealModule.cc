#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <Validation/EcalDigis/interface/EcalDigisValidation.h>
DEFINE_FWK_MODULE(EcalDigisValidation);

#include <Validation/EcalDigis/interface/EcalBarrelDigisValidation.h>
DEFINE_FWK_MODULE(EcalBarrelDigisValidation);

#include <Validation/EcalDigis/interface/EcalEndcapDigisValidation.h>
DEFINE_FWK_MODULE(EcalEndcapDigisValidation);

#include <Validation/EcalDigis/interface/EcalPreshowerDigisValidation.h>
DEFINE_FWK_MODULE(EcalPreshowerDigisValidation);

#include <Validation/EcalDigis/interface/EcalPreshowerNoiseDistrib.h>
DEFINE_FWK_MODULE(EcalPreshowerNoiseDistrib);

#include <Validation/EcalDigis/interface/EcalMixingModuleValidation.h>
DEFINE_FWK_MODULE(EcalMixingModuleValidation);

#include "Validation/EcalDigis/interface/EcalSelectiveReadoutValidation.h"
DEFINE_FWK_MODULE(EcalSelectiveReadoutValidation);
