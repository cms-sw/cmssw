import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi import *
from RecoTracker.TkSeedingLayers.MixedLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
#from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *

import RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi
globalSeedsFromSiStripTriplets = RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi.seedGeneratorFromRegionHitsEDProducer.clone(
    OrderedHitsFactoryPSet = cms.PSet(
      ComponentName = cms.string('StandardHitTripletGenerator'),
      SeedingLayers = cms.InputTag('SiStripLayerTriplets'),
      RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
      ),
     ),
     SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsTripletOnlyCreator)
)
