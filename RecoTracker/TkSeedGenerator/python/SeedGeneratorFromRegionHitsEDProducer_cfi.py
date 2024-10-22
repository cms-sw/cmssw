import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi import *
#from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsTripletOnlyCreator_cfi import *

seedGeneratorFromRegionHitsEDProducer = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string(''),
        SeedingLayers = cms.InputTag(''),
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
                 MaxNumberOfStripClusters = cms.uint32(400000),
                 ClusterCollectionLabel = cms.InputTag("siStripClusters"),
                 MaxNumberOfPixelClusters = cms.uint32(40000),
                 PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
                 cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)")
                 ),
)

# Disable too many clusters check until we have an updated cut string for phase1
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(seedGeneratorFromRegionHitsEDProducer, # FIXME
    ClusterCheckPSet = dict(doClusterCheck = False)
)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
peripheralPbPb.toModify(seedGeneratorFromRegionHitsEDProducer,
                        ClusterCheckPSet = dict(doClusterCheck = True,  # FIXMETOO
                                                cut = "strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
                        )

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(seedGeneratorFromRegionHitsEDProducer,
               ClusterCheckPSet = dict(doClusterCheck = True, # FIXMETOO
                                       cut = "strip < 1000000 && pixel < 100000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/2.)",
                                       MaxNumberOfPixelClusters = 100000)
               )

