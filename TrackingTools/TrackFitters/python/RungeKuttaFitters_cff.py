import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
RKTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone()
RKTrajectoryFitter.ComponentName = 'RKFitter'
RKTrajectoryFitter.Propagator = 'RungeKuttaTrackerPropagator'


import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
RKTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone()
RKTrajectorySmoother.ComponentName = 'RKSmoother'
RKTrajectorySmoother.Propagator = 'RungeKuttaTrackerPropagator'


import TrackingTools.TrackFitters.KFFittingSmoother_cfi
RKFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone()
RKFittingSmoother.ComponentName = 'RKFittingSmoother'
RKFittingSmoother.Fitter = 'RKFitter'
RKFittingSmoother.Smoother = 'RKSmoother'


KFFittingSmootherWithOutliersRejectionAndRK = RKFittingSmoother.clone()
KFFittingSmootherWithOutliersRejectionAndRK.ComponentName = 'KFFittingSmootherWithOutliersRejectionAndRK'
KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut = 20.0                
KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits = 3

