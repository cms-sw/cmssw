import FWCore.ParameterSet.Config as cms

GsfTrackRefitter = cms.EDFilter("GsfTrackRefitter",
    src = cms.InputTag("pixelMatchGsfFit"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdGsfElectronPropagator')
)


