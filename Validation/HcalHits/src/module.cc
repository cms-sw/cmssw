#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "Validation/HcalHits/interface/SimG4HcalValidation.h"

DEFINE_SIMWATCHER(SimG4HcalValidation);

#include "Validation/HcalHits/interface/HcalSimHitStudy.h"
#include "Validation/HcalHits/interface/ZdcSimHitStudy.h"

DEFINE_FWK_MODULE(HcalSimHitStudy);
DEFINE_FWK_MODULE(ZdcSimHitStudy);
