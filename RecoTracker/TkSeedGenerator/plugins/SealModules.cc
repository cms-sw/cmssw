#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SeedGeneratorFromRegionHitsEDProducer.h"
#include "SeedGeneratorFromProtoTracksEDProducer.h"
#include "SeedCombiner.h"

DEFINE_FWK_MODULE(SeedGeneratorFromRegionHitsEDProducer);
DEFINE_FWK_MODULE(SeedGeneratorFromProtoTracksEDProducer);
DEFINE_FWK_MODULE(SeedCombiner);


#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "SeedFromConsecutiveHitsCreator.h"
#include "SeedFromConsecutiveHitsStraightLineCreator.h"
#include "SeedFromConsecutiveHitsTripletOnlyCreator.h"

DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsCreator, "SeedFromConsecutiveHitsCreator");
DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsStraightLineCreator, "SeedFromConsecutiveHitsStraightLineCreator");
DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsTripletOnlyCreator, "SeedFromConsecutiveHitsTripletOnlyCreator");
