import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
MRHTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone()
MRHTrajectorySmoother.ComponentName = 'MRHSmoother'
MRHTrajectorySmoother.Estimator = 'MRHChi2'
MRHTrajectorySmoother.Propagator = 'RungeKuttaTrackerPropagator'

