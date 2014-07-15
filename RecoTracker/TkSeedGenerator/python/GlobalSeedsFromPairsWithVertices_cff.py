import FWCore.ParameterSet.Config as cms


from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
from RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi import *

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionWithVertices_cfi import *
import RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi
globalSeedsFromPairsWithVertices = RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi.seedGeneratorFromRegionHitsEDProducer.clone(
    OrderedHitsFactoryPSet = cms.PSet(
      ComponentName = cms.string('StandardHitPairGenerator'),
      SeedingLayers = cms.InputTag('MixedLayerPairs'),
      maxElement = cms.uint32(1000000)
    ),
    RegionFactoryPSet = cms.PSet(
      RegionPSetWithVerticesBlock,
      ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer')
    )
)    

