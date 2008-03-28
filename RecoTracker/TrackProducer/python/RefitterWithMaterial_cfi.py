import FWCore.ParameterSet.Config as cms

TrackRefitter = cms.EDFilter("TrackRefitter",
    src = cms.InputTag("generalTracks"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string(''),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


