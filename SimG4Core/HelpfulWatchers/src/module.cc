#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/HelpfulWatchers/src/SimTracer.h"
#include "SimG4Core/HelpfulWatchers/src/BeginOfTrackCounter.h"

#include "PluginManager/ModuleDef.h"

using namespace simwatcher;
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER (SimTracer);
DEFINE_SIMWATCHER (BeginOfTrackCounter);

