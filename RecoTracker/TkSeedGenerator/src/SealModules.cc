#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SeedGeneratorFromRegionHitsEDProducer.h"
#include "SeedGeneratorFromProtoTracksEDProducer.h"

#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelSeedGeneratorWithVertex.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromRectangularTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/GlobalPixelLessSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/GlobalMixedSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/CosmicSeedGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromTrk.h"
#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromCandidate.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromRegionHitsEDProducer);
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromProtoTracksEDProducer);
DEFINE_ANOTHER_FWK_MODULE(GlobalPixelSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(GlobalPixelSeedGeneratorWithVertex);
DEFINE_ANOTHER_FWK_MODULE(RegionalPixelSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(RegionalPixelSeedGeneratorFromCandidate);
DEFINE_ANOTHER_FWK_MODULE(GlobalPixelLessSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(GlobalMixedSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(CosmicSeedGenerator);
DEFINE_ANOTHER_FWK_MODULE(RegionalPixelSeedGeneratorFromTrk);

