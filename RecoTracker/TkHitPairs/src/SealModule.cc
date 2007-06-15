#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"

DEFINE_SEAL_PLUGIN(OrderedHitsGeneratorFactory, CombinedHitPairGenerator, "StandardHitPairGenerator");

