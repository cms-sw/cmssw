import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.dafTrackProducer_cfi import dafTrackProducer
TracksDAF = dafTrackProducer.clone(
    src = "DAFTrackCandidateMaker",
    UpdatorName = 'SiTrackerMultiRecHitUpdator',
    beamSpot = "offlineBeamSpot",
    Fitter = 'MRHFittingSmoother',
    MeasurementCollector = 'simpleMultiRecHitCollector',
    NavigationSchool = '',
    MeasurementTrackerEvent = 'MeasurementTrackerEvent',
    TrajectoryInEvent = False,
    TTRHBuilder = 'WithAngleAndTemplate',
    Propagator = 'RungeKuttaTrackerPropagator',
    MinHits = 3,
    TrajAnnealingSaving = False 
)


