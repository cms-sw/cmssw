import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
myTTRHBuilderWithoutAngle4PixelTriplets = copy.deepcopy(ttrhbwr)
myTTRHBuilderWithoutAngle4PixelTriplets.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle4PixelTriplets.ComponentName = 'TTRHBuilderWithoutAngle4PixelTriplets'

