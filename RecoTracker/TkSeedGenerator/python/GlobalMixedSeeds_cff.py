import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
# initialize geometry #####################

# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# generate pixel seeds #####################
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi import *
from RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi import *


import RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi
globalMixedSeeds = RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi.seedGeneratorFromRegionHitsEDProducer.clone(
    OrderedHitsFactoryPSet = cms.PSet(
       ComponentName = cms.string('StandardHitPairGenerator'),
       SeedingLayers = cms.string('MixedLayerPairs'),
       maxElement = cms.uint32(1000000)
    )
)
