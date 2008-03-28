import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
DAFTrajectorySmoother = copy.deepcopy(KFTrajectorySmoother)
DAFTrajectorySmoother.ComponentName = 'DAFSmoother'
DAFTrajectorySmoother.Estimator = 'MRHChi2'
DAFTrajectorySmoother.Propagator = 'RungeKuttaTrackerPropagator'

