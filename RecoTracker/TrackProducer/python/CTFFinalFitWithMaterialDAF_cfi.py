import FWCore.ParameterSet.Config as cms

TracksDAF = cms.EDProducer("DAFTrackProducer",
    src = cms.InputTag("DAFTrackCandidateMaker"),
    UpdatorName = cms.string('SiTrackerMultiRecHitUpdator'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('MRHFittingSmoother'),
    MeasurementCollector = cms.string('simpleMultiRecHitCollector'),
    NavigationSchool = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    MinHits = cms.int32(3),
    TrajAnnealingSaving = cms.bool(False) 
)


