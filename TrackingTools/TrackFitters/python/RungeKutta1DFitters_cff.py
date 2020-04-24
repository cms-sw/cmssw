import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
RK1DTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('RK1DFitter'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFSwitching1DUpdator')
)

import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
RK1DTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = cms.string('RK1DSmoother'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFSwitching1DUpdator')
)

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
RK1DFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = cms.string('RK1DFittingSmoother'),
    Fitter = cms.string('RK1DFitter'),
    Smoother = cms.string('RK1DSmoother')
)

RKOutliers1DFittingSmoother = RK1DFittingSmoother.clone(
    ComponentName = cms.string('RKOutliers1DFittingSmoother'),
    EstimateCut = cms.double(20.0),
    MinNumberOfHits = cms.int32(3),
)
