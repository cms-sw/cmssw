
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrackFinder.h"
using cms::CosmicTrackFinder;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CosmicTrackFinder);
