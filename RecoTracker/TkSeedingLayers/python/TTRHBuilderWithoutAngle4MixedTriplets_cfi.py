import FWCore.ParameterSet.Config as cms

import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi 
myTTRHBuilderWithoutAngle4MixedTriplets = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE      = 'Fake',
    ComponentName = 'TTRHBuilderWithoutAngle4MixedTriplets'
)
