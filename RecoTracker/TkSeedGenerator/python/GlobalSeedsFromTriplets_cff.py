import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi import *
from RecoTracker.TkSeedingLayers.MixedLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.PixelSeeding.PixelTripletHLTGenerator_cfi import *
#from RecoTracker.PixelSeeding.PixelTripletLargeTipGenerator_cfi import *

import RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi
globalSeedsFromTriplets = RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi.seedGeneratorFromRegionHitsEDProducer.clone(
    OrderedHitsFactoryPSet = dict(
      ComponentName = 'StandardHitTripletGenerator',
      SeedingLayers = 'PixelLayerTriplets',
      GeneratorPSet = cms.PSet(PixelTripletHLTGenerator.clone(maxElement = cms.uint32(1000000)))
# this one uses an exact helix extrapolation and can deal correctly with
# arbitrarily large D0 and generally exhibits a smaller fake rate:
#     GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
    )
)
