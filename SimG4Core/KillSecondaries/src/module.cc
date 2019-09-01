#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/KillSecondaries/interface/KillSecondariesRunAction.h"
#include "SimG4Core/KillSecondaries/interface/KillSecondariesTrackAction.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(KillSecondariesRunAction);
DEFINE_SIMWATCHER(KillSecondariesTrackAction);
