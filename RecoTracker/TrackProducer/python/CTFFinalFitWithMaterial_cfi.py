import FWCore.ParameterSet.Config as cms

ctfWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


