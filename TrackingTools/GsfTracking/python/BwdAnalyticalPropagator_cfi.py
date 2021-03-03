import FWCore.ParameterSet.Config as cms

#import copy
from TrackingTools.GsfTracking.FwdAnalyticalPropagator_cfi import *
#
# "backward" propagator for electrons
#
bwdAnalyticalPropagator = fwdAnalyticalPropagator.clone(
    ComponentName = 'bwdAnalyticalPropagator',
    PropagationDirection = 'oppositeToMomentum'
)
