import FWCore.ParameterSet.Config as cms


from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.PixelTrackFitting.pixelTracks_cfi import pixelTracks
from RecoTracker.TkSeedGenerator.SeedGeneratorFromProtoTracksEDProducer_cfi import *
pixelTTRHBuilderWithoutAngle = ttrhbwr.clone(
    StripCPE      = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngle'
)
