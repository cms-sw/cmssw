import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
myTTRHBuilderWithoutAngle4PixelPairs = copy.deepcopy(ttrhbwr)
myTTRHBuilderWithoutAngle4PixelPairs.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle4PixelPairs.ComponentName = 'TTRHBuilderWithoutAngle4PixelPairs'

