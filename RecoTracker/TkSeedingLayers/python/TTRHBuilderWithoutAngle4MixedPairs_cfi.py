import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
myTTRHBuilderWithoutAngle4MixedPairs = copy.deepcopy(ttrhbwr)
myTTRHBuilderWithoutAngle4MixedPairs.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle4MixedPairs.ComponentName = 'TTRHBuilderWithoutAngle4MixedPairs'

