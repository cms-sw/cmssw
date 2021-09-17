import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
RK1DTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'RK1DFitter',
    Propagator    = 'RungeKuttaTrackerPropagator',
    Updator       = 'KFSwitching1DUpdator'
)

import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
RK1DTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'RK1DSmoother',
    Propagator    = 'RungeKuttaTrackerPropagator',
    Updator       = 'KFSwitching1DUpdator'
)

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
RK1DFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'RK1DFittingSmoother',
    Fitter        = 'RK1DFitter',
    Smoother      = 'RK1DSmoother'
)

RKOutliers1DFittingSmoother = RK1DFittingSmoother.clone(
    ComponentName   = 'RKOutliers1DFittingSmoother',
    EstimateCut     = 20.0,
    MinNumberOfHits = 3,
)
