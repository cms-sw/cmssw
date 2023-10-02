import FWCore.ParameterSet.Config as cms


from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
#from RecoTracker.TkSeedingLayers.PixelLessLayerPairs_cfi import *
from RecoTracker.TkSeedingLayers.PixelLessLayerPairs4PixelLessTracking_cfi import *

import RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi
globalPixelLessSeeds = RecoTracker.TkSeedGenerator.SeedGeneratorFromRegionHitsEDProducer_cfi.seedGeneratorFromRegionHitsEDProducer.clone(
    OrderedHitsFactoryPSet = dict(
        ComponentName = 'StandardHitPairGenerator',
        SeedingLayers = 'pixelLessLayerPairs4PixelLessTracking',
        maxElement    = 100000
    ),
    ## whatever happens to the beam spot
    RegionFactoryPSet = dict(RegionPSet = dict(originHalfLength = 40)),
    ## safe against APV-induced noise
    ClusterCheckPSet = dict(MaxNumberOfStripClusters = 5000)
)
