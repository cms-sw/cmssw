import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
DAFTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone()
DAFTrajectoryFitter.ComponentName = 'DAFFitter'
DAFTrajectoryFitter.Estimator = 'MRHChi2'
DAFTrajectoryFitter.Propagator = 'RungeKuttaTrackerPropagator'

