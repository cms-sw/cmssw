import FWCore.ParameterSet.Config as cms

GsfTrackRefitter = cms.EDProducer("GsfTrackRefitter",
    src = cms.InputTag("pixelMatchGsfFit"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdGsfElectronPropagator'),
    constraint = cms.string(''),
    #set to "", the secondary hit pattern will not be filled (backward compatible with DetLayer=0)                               
    NavigationSchool = cms.string(''),
    MeasurementTracker = cms.string(''),

    AlgorithmName = cms.string('gsf')
)


