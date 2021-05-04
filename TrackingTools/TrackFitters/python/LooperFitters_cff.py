import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
LooperTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'LooperFitter',
    Propagator    = 'PropagatorWithMaterialForLoopers'
)

import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
LooperTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName  = 'LooperSmoother',
    Propagator     = 'PropagatorWithMaterialForLoopers',
    errorRescaling = 10.0,
)

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
LooperFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'LooperFittingSmoother',
    Fitter        = 'LooperFitter',
    Smoother      = 'LooperSmoother',
    EstimateCut   = 20.0,
    # ggiurgiu@fnal.gov : Any value lower than -15 turns off this cut.
    # Recommended default value: -14.0. This will reject only the worst hits with negligible loss in track efficiency.  
    LogPixelProbabilityCut = -14.0,
    MinNumberOfHits = 3
)
