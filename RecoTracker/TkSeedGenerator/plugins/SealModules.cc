#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SeedGeneratorFromRegionHitsEDProducer.h"
#include "SeedGeneratorFromProtoTracksEDProducer.h"
#include "SeedCombiner.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromRegionHitsEDProducer);
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromProtoTracksEDProducer);
DEFINE_ANOTHER_FWK_MODULE(SeedCombiner);

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHitsStrightLineCreator.h"


DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsCreator, "SeedFromConsecutiveHitsCreator");
DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsStrightLineCreator, "SeedFromConsecutiveHitsStrightLineCreator");

