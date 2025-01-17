import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.trackProducer_cfi import trackProducer
TrackProducer = trackProducer.clone(
    useSimpleMF = False,
    SimpleMagneticField = "", # only if useSimpleMF is True
    src = "ckfTrackCandidates",
    clusterRemovalInfo = "",
    beamSpot = "offlineBeamSpot",
    Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK',
    useHitsSplitting = False,
    TrajectoryInEvent = False,
    TTRHBuilder = 'WithAngleAndTemplate',
    AlgorithmName = 'undefAlgorithm',
    Propagator = 'RungeKuttaTrackerPropagator',

    # this parameter decides if the propagation to the beam line
    # for the track parameters defiition is from the first hit
    # or from the closest to the beam line
    # true for cosmics/beam halo, false for collision tracks (needed by loopers)
    GeometricInnerState = False,

    ### These are paremeters related to the filling of the Secondary hit-patterns
    #set to "", the secondary hit pattern will not be filled (backward compatible with DetLayer=0)
    NavigationSchool = 'SimpleNavigationSchool',
    MeasurementTracker = '',
    MeasurementTrackerEvent = 'MeasurementTrackerEvent'
)
