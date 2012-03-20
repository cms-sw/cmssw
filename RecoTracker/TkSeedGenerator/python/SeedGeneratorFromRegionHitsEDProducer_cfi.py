import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsTripletOnlyCreator_cfi import *

seedGeneratorFromRegionHitsEDProducer = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string(''),
        SeedingLayers = cms.string(''),
        maxElement = cms.uint32(100000)
    ),
    SeedMergerPSet = cms.PSet(
        # layer list for the merger, as defined in (or modified from):
        # RecoPixelVertexing/PixelTriplets/python/quadrupletseedmerging_cff.py
        layerListName = cms.string( "PixelSeedMergerQuadruplets" ),
        # merge triplets -> quadruplets if applicable?
        mergeTriplets = cms.bool( True ),
        # add remaining (non-merged) triplets to merged output quadruplets?
        # (results in a "mixed" output)
        addRemainingTriplets = cms.bool( False ),
        # the builder
        ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" )
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
                 MaxNumberOfCosmicClusters = cms.uint32(50000),
                 ClusterCollectionLabel = cms.InputTag("siStripClusters"),
                 MaxNumberOfPixelClusters = cms.uint32(10000),
                 PixelClusterCollectionLabel = cms.InputTag("siPixelClusters")
                 ),
)
