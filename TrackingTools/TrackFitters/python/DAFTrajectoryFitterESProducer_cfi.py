import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
DAFTrajectoryFitter = copy.deepcopy(KFTrajectoryFitter)
DAFTrajectoryFitter.ComponentName = 'DAFFitter'
DAFTrajectoryFitter.Estimator = 'MRHChi2'
DAFTrajectoryFitter.Propagator = 'RungeKuttaTrackerPropagator'

