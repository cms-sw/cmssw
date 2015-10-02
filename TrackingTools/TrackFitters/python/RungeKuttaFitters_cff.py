import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
RKTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone()
RKTrajectoryFitter.ComponentName = 'RKFitter'
RKTrajectoryFitter.Propagator = 'RungeKuttaTrackerPropagator'


import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
RKTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone()
RKTrajectorySmoother.ComponentName = 'RKSmoother'
RKTrajectorySmoother.Propagator = 'RungeKuttaTrackerPropagator'


import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
RKFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
RKFittingSmoother.ComponentName = 'RKFittingSmoother'
RKFittingSmoother.Fitter = 'RKFitter'
RKFittingSmoother.Smoother = 'RKSmoother'


KFFittingSmootherWithOutliersRejectionAndRK = RKFittingSmoother.clone()
KFFittingSmootherWithOutliersRejectionAndRK.ComponentName = 'KFFittingSmootherWithOutliersRejectionAndRK'
KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut = 20.0                
KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits = 3

