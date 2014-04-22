import FWCore.ParameterSet.Config as cms

TrackProducer = cms.EDProducer("TrackProducer",
    useSimpleMF = cms.bool(False),
    SimpleMagneticField = cms.string(""),
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

    # this parameter decides if the propagation to the beam line
    # for the track parameters defiition is from the first hit
    # or from the closest to the beam line
    # true for cosmics/beam halo, false for collision tracks (needed by loopers)
    GeometricInnerState = cms.bool(False),

    ### These are paremeters related to the filling of the Secondary hit-patterns                               
    #set to "", the secondary hit pattern will not be filled (backward compatible with DetLayer=0)    
    NavigationSchool = cms.string('SimpleNavigationSchool'),          
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent'),                   
)



