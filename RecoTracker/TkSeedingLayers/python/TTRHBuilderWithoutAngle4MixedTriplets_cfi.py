import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
myTTRHBuilderWithoutAngle4MixedTriplets = copy.deepcopy(ttrhbwr)
myTTRHBuilderWithoutAngle4MixedTriplets.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle4MixedTriplets.ComponentName = 'TTRHBuilderWithoutAngle4MixedTriplets'

