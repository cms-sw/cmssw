import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
RKTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('RKFitter'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
RKTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone(
    ComponentName = cms.string('RKSmoother'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
RKFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone(
    ComponentName = cms.string('RKFittingSmoother'),
    Fitter = cms.string('RKFitter'),
    Smoother = cms.string('RKSmoother')
)

KFFittingSmootherWithOutliersRejectionAndRK = RKFittingSmoother.clone(
    ComponentName = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    EstimateCut = cms.double(20.0),
    # ggiurgiu@fnal.gov : Any value lower than -15 turns off this cut.
    # Recommended default value: -14.0. This will reject only the worst hits with negligible loss in track efficiency.  
    LogPixelProbabilityCut = cms.double(-14.0),                               
    MinNumberOfHits = cms.int32(3)
)
