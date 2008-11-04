import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
MRHTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone()
MRHTrajectoryFitter.ComponentName = 'MRHFitter'
MRHTrajectoryFitter.Estimator = 'MRHChi2'
MRHTrajectoryFitter.Propagator = 'RungeKuttaTrackerPropagator'

