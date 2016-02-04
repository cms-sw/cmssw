import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
MRHTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'MRHFitter',
    Estimator = 'MRHChi2',
    Propagator = 'RungeKuttaTrackerPropagator'
    )
import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
MRHTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'MRHSmoother',
    Estimator = 'MRHChi2',
    Propagator = 'RungeKuttaTrackerPropagator'
    )

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
MRHFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone(
    ComponentName = 'MRHFittingSmoother',
    Fitter = 'MRHFitter',
    Smoother = 'MRHSmoother'
    )

