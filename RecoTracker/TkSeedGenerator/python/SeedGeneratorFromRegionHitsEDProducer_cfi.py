import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsTripletOnlyCreator_cfi import *

seedGeneratorFromRegionHitsEDProducer = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string(''),
        SeedingLayers = cms.string('')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
# This works best ...
    SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsCreator)
# except for large impact parameter pixel-pair seeding, when this is better ...
#   SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsStraightLineCreator)                                                       
# and this one respectively for large-D0 triplets:
#   SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsTripletOnlyCreator)
)
