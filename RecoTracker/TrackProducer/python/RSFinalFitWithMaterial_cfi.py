import FWCore.ParameterSet.Config as cms

rsWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("rsTrackCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('rs'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


