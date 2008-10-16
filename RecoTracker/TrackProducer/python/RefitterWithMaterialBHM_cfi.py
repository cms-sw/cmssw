import FWCore.ParameterSet.Config as cms

TrackRefitter = cms.EDFilter("TrackRefitter",
    src = cms.InputTag("ctfWithMaterialTracksBeamHaloMuon"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('KFFittingSmootherBH'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('BeamHaloPropagatorAlong'),
 
    ### fitting without constraints
    constraint = cms.string(''),

    ### fitting with constraints                             
    #constraint = cms.string('momentum'),
    #constraint = cms.string('vertex'),

    ### Usually this parameter has not to be set True because 
    ### matched hits in the Tracker are already split when 
    ### the tracks are reconstructed the first time                         
    useHitsSplitting = cms.bool(False),


    TrajectoryInEvent = cms.bool(True)
)


