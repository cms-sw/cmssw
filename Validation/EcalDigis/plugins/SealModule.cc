#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EcalDigisValidation.h"
DEFINE_FWK_MODULE(EcalDigisValidation);

#include "EcalBarrelDigisValidation.h"
DEFINE_FWK_MODULE(EcalBarrelDigisValidation);

#include "EcalEndcapDigisValidation.h"
DEFINE_FWK_MODULE(EcalEndcapDigisValidation);

#include "EcalPreshowerDigisValidation.h"
DEFINE_FWK_MODULE(EcalPreshowerDigisValidation);

#include "EcalPreshowerNoiseDistrib.h"
DEFINE_FWK_MODULE(EcalPreshowerNoiseDistrib);

#include "EcalMixingModuleValidation.h"
DEFINE_FWK_MODULE(EcalMixingModuleValidation);

#include "EcalSelectiveReadoutValidation.h"
DEFINE_FWK_MODULE(EcalSelectiveReadoutValidation);
