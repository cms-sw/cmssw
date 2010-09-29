import FWCore.ParameterSet.Config as cms

TrackProducer = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),

    ### These are paremeters related to the filling of the Secondary hit-patterns                               
    #set to "", the secondary hit pattern will not be filled (backward compatible with DetLayer=0)    
    NavigationSchool = cms.string('SimpleNavigationSchool'),          
    MeasurementTracker = cms.string('')                   
)



