
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchHelixMaker.h"
#include "RecoTracker/RoadSearchHelixMaker/interface/RoadSearchTrackListCleaner.h"
#include "RecoTracker/RoadSearchHelixMaker/interface/TrackListMerger.h"

using cms::RoadSearchHelixMaker;
using cms::RoadSearchTrackListCleaner;
using cms::TrackListMerger;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchHelixMaker);
DEFINE_ANOTHER_FWK_MODULE(RoadSearchTrackListCleaner);
DEFINE_ANOTHER_FWK_MODULE(TrackListMerger);
