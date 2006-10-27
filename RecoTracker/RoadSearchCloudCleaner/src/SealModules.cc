
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchCloudCleaner/interface/RoadSearchCloudCleaner.h"

using cms::RoadSearchCloudCleaner;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchCloudCleaner);
