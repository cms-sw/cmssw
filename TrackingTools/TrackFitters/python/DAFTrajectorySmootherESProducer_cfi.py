import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
DAFTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone()
DAFTrajectorySmoother.ComponentName = 'DAFSmoother'
DAFTrajectorySmoother.Estimator = 'MRHChi2'
DAFTrajectorySmoother.Propagator = 'RungeKuttaTrackerPropagator'

