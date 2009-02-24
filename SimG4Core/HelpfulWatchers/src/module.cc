#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/HelpfulWatchers/src/SimTracer.h"
#include "SimG4Core/HelpfulWatchers/src/BeginOfTrackCounter.h"
//Adding a Watcher to collect G4step statistics:
#include "SimG4Core/HelpfulWatchers/src/G4StepStatistics.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

using namespace simwatcher;
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER (SimTracer);
DEFINE_SIMWATCHER (BeginOfTrackCounter);

//Adding a Watcher to collect G4step statistics:
DEFINE_SIMWATCHER (G4StepStatistics);
