import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.GsfTracking.FwdAnalyticalPropagator_cfi import *
#
# "backward" propagator for electrons
#
bwdAnalyticalPropagator = copy.deepcopy(fwdAnalyticalPropagator)
bwdAnalyticalPropagator.ComponentName = 'bwdAnalyticalPropagator'
bwdAnalyticalPropagator.PropagationDirection = 'oppositeToMomentum'

