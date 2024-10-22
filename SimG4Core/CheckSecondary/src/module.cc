#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/CheckSecondary/interface/CheckSecondary.h"
#include "SimG4Core/CheckSecondary/interface/StoreSecondary.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(CheckSecondary);
DEFINE_SIMWATCHER(StoreSecondary);
