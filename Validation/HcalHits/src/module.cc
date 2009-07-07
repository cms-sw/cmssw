#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "Validation/HcalHits/interface/SimG4HcalValidation.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER (SimG4HcalValidation);

#include "Validation/HcalHits/interface/HcalSimHitStudy.h"

DEFINE_ANOTHER_FWK_MODULE (HcalSimHitStudy);
