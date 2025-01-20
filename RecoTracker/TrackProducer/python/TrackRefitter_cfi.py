import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.trackRefitter_cfi import trackRefitter
TrackRefitter = trackRefitter.clone(
    src = "generalTracks",
    beamSpot = "offlineBeamSpot",
    Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK',
    TTRHBuilder = 'WithAngleAndTemplate',
    AlgorithmName = 'undefAlgorithm',
    Propagator = 'RungeKuttaTrackerPropagator',

    ### fitting without constraints
    constraint = '',
    srcConstr  = '',

    ### fitting with constraints
    #constraint = 'momentum',
    #constraint = 'vertex',

    ### Usually this parameter has not to be set True because
    ### matched hits in the Tracker are already split when
    ### the tracks are reconstructed the first time
    useHitsSplitting = False,
    TrajectoryInEvent = True,

    # this parameter decides if the propagation to the beam line
    # for the track parameters defiition is from the first hit
    # or from the closest to the beam line
    # true for cosmics, false for collision tracks (needed by loopers)
    GeometricInnerState = False,

    # Navigation school is necessary to fill the secondary hit patterns
    NavigationSchool = 'SimpleNavigationSchool',
    MeasurementTracker = '',
    MeasurementTrackerEvent = 'MeasurementTrackerEvent',
    #
    # in order to avoid to fill the secondary hit patterns and
    # refit the tracks more quickly 
    #NavigationSchool = ''
)


