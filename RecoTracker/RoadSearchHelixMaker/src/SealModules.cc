
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMaker.h"
#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchTrackListCleaner.h"

using cms::RoadSearchHelixMaker;
using cms::RoadSearchTrackListCleaner;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchHelixMaker);
DEFINE_ANOTHER_FWK_MODULE(RoadSearchTrackListCleaner);
