#include "SimG4Core/KillSecondaries/interface/KillSecondariesRunAction.h"
#include "SimG4Core/KillSecondaries/interface/KillSecondariesTrackAction.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "PluginManager/ModuleDef.h"
    
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER(KillSecondariesRunAction);
DEFINE_SIMWATCHER(KillSecondariesTrackAction);
