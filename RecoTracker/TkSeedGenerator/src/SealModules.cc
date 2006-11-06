#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromRectangularTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelLessSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/GlobalMixedSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/CosmicSeedGenerator.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GlobalPixelSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(RegionalPixelSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(RegionalPixelSeedGeneratorFromRectangularTrackingRegion);
DEFINE_ANOTHER_FWK_MODULE(GlobalPixelLessSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(GlobalMixedSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(CosmicSeedGenerator);
