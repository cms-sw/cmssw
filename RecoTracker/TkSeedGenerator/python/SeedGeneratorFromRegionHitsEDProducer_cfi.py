import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsTripletOnlyCreator_cfi import *

seedGeneratorFromRegionHitsEDProducer = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string(''),
        SeedingLayers = cms.string(''),
        maxElement = cms.uint32(1000000)
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPsetFomBeamSpotBlockFixedZ,
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
# This works best ...
    SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsCreator),
# except for large impact parameter pixel-pair seeding, when this is better ...
#   SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsStraightLineCreator)                                                       
# and this one respectively for large-D0 triplets:
#   SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsTripletOnlyCreator)

##add a protection for too many clusters in the event.
ClusterCheckPSet = cms.PSet(
                 doClusterCheck = cms.bool(True),
                 MaxNumberOfCosmicClusters = cms.uint32(400000),
                 ClusterCollectionLabel = cms.InputTag("siStripClusters"),
                 MaxNumberOfPixelClusters = cms.uint32(40000),
                 PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
                 cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)")
                 ),
)
