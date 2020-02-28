#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "Validation/EcalHits/interface/EcalSimHitsValidProducer.h"
DEFINE_SIMWATCHER(EcalSimHitsValidProducer);

#include <Validation/EcalHits/interface/EcalSimHitsValidation.h>
DEFINE_FWK_MODULE(EcalSimHitsValidation);

#include <Validation/EcalHits/interface/EcalBarrelSimHitsValidation.h>
DEFINE_FWK_MODULE(EcalBarrelSimHitsValidation);

#include <Validation/EcalHits/interface/EcalEndcapSimHitsValidation.h>
DEFINE_FWK_MODULE(EcalEndcapSimHitsValidation);

#include <Validation/EcalHits/interface/EcalPreshowerSimHitsValidation.h>
DEFINE_FWK_MODULE(EcalPreshowerSimHitsValidation);
