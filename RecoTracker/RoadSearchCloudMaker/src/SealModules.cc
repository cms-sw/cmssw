
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMaker.h"
#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudCleaner.h"

using cms::RoadSearchCloudMaker;
using cms::RoadSearchCloudCleaner;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchCloudMaker)
DEFINE_ANOTHER_FWK_MODULE(RoadSearchCloudCleaner)
