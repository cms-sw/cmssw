import FWCore.ParameterSet.Config as cms

ctfWithMaterialTracksDAF = cms.EDProducer("DAFTrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    UpdatorName = cms.string('SiTrackerMultiRecHitUpdator'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('DAFFittingSmoother'),
    MeasurementCollector = cms.string('groupedMultiRecHitCollector'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('PropagatorWithMaterial'),
    MinHits = cms.int32(5)
)


