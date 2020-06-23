#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SeedGeneratorFromRegionHitsEDProducer.h"
#include "SeedGeneratorFromProtoTracksEDProducer.h"
#include "SeedGeneratorFromTTracksEDProducer.h"
#include "SeedCombiner.h"

DEFINE_FWK_MODULE(SeedGeneratorFromRegionHitsEDProducer);
DEFINE_FWK_MODULE(SeedGeneratorFromProtoTracksEDProducer);
DEFINE_FWK_MODULE(SeedGeneratorFromTTracksEDProducer);
DEFINE_FWK_MODULE(SeedCombiner);

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "SeedFromConsecutiveHitsCreator.h"
#include "SeedFromConsecutiveHitsStraightLineCreator.h"
#include "SeedFromConsecutiveHitsTripletOnlyCreator.h"

DEFINE_EDM_PLUGIN(SeedCreatorFactory, SeedFromConsecutiveHitsCreator, "SeedFromConsecutiveHitsCreator");
DEFINE_EDM_PLUGIN(SeedCreatorFactory,
                  SeedFromConsecutiveHitsStraightLineCreator,
                  "SeedFromConsecutiveHitsStraightLineCreator");
DEFINE_EDM_PLUGIN(SeedCreatorFactory,
                  SeedFromConsecutiveHitsTripletOnlyCreator,
                  "SeedFromConsecutiveHitsTripletOnlyCreator");

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "CombinedMultiHitGenerator.h"
DEFINE_EDM_PLUGIN(OrderedHitsGeneratorFactory, CombinedMultiHitGenerator, "StandardMultiHitGenerator");

#include "MultiHitGeneratorFromChi2.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
DEFINE_EDM_PLUGIN(MultiHitGeneratorFromPairAndLayersFactory, MultiHitGeneratorFromChi2, "MultiHitGeneratorFromChi2");

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFromRegionHitsEDProducerT.h"
using SeedCreatorFromRegionConsecutiveHitsEDProducer =
    SeedCreatorFromRegionHitsEDProducerT<SeedFromConsecutiveHitsCreator>;
DEFINE_FWK_MODULE(SeedCreatorFromRegionConsecutiveHitsEDProducer);

using SeedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer =
    SeedCreatorFromRegionHitsEDProducerT<SeedFromConsecutiveHitsTripletOnlyCreator>;
DEFINE_FWK_MODULE(SeedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer);
