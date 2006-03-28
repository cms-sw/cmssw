
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMaker.h"

using cms::RoadSearchCloudMaker;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RoadSearchCloudMaker)
