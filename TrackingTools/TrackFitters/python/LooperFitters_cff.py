import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
LooperTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('LooperFitter'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers')
)

import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
LooperTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone(
    ComponentName = cms.string('LooperSmoother'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    errorRescaling = cms.double(10.0),
)

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
LooperFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone(
    ComponentName = cms.string('LooperFittingSmoother'),
    Fitter = cms.string('LooperFitter'),
    Smoother = cms.string('LooperSmoother'),
    EstimateCut = cms.double(20.0),
    # ggiurgiu@fnal.gov : Any value lower than -15 turns off this cut.
    # Recommended default value: -14.0. This will reject only the worst hits with negligible loss in track efficiency.  
    LogPixelProbabilityCut = cms.double(-14.0),                               
    MinNumberOfHits = cms.int32(3)
)

