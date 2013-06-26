import FWCore.ParameterSet.Config as cms

TrackRefitter = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("generalTracks"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),

    ### fitting without constraints
    constraint = cms.string(''),
    srcConstr  = cms.InputTag(''),                   

    ### fitting with constraints                             
    #constraint = cms.string('momentum'),
    #constraint = cms.string('vertex'),

    ### Usually this parameter has not to be set True because 
    ### matched hits in the Tracker are already split when 
    ### the tracks are reconstructed the first time                         
    useHitsSplitting = cms.bool(False),

    TrajectoryInEvent = cms.bool(True),

    # this parameter decides if the propagation to the beam line
    # for the track parameters defiition is from the first hit
    # or from the closest to the beam line
    # true for cosmics, false for collision tracks (needed by loopers)
    GeometricInnerState = cms.bool(False),

    # Navigation school is necessary to fill the secondary hit patterns                         
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    MeasurementTracker = cms.string(''),                                              
    #
    # in order to avoid to fill the secondary hit patterns and
    # refit the tracks more quickly 
    #NavigationSchool = cms.string('') 
)


