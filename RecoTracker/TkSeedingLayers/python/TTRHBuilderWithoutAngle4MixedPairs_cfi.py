import FWCore.ParameterSet.Config as cms

import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi 
myTTRHBuilderWithoutAngle4MixedPairs = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE      = 'Fake',
    ComponentName = 'TTRHBuilderWithoutAngle4MixedPairs'
)
