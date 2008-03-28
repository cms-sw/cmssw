import FWCore.ParameterSet.Config as cms

ctfAnalyticalTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('KFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('AnalyticalPropagator')
)


