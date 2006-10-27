
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchHelixMaker/test/RoadSearchTrackReader.h"

using cms::RoadSearchTrackReader;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchTrackReader);

