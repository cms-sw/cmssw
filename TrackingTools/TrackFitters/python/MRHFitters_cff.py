import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.MRHChi2MeasurementEstimatorESProducer_cfi import *

import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
MRHTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'MRHFitter',
    Estimator     = 'MRHChi2',
    Propagator    = 'RungeKuttaTrackerPropagator'
)
import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
MRHTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'MRHSmoother',
    Estimator     = 'MRHChi2',
    Propagator    = 'RungeKuttaTrackerPropagator'
)

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
MRHFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'MRHFittingSmoother',
    Fitter        = 'MRHFitter',
    Smoother      = 'MRHSmoother'
)
