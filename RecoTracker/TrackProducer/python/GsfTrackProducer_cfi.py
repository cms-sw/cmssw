import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.gsfTrackProducer_cfi import gsfTrackProducer
gsfTrackProducer = gsfTrackProducer.clone(
    src = "CkfElectronCandidates",
    beamSpot = "offlineBeamSpot",
    Fitter = 'GsfElectronFittingSmoother',
    useHitsSplitting = False,
    TrajectoryInEvent = False,
    TTRHBuilder = 'WithTrackAngle',
    Propagator = 'fwdElectronPropagator',
    NavigationSchool = 'SimpleNavigationSchool',
    MeasurementTracker = '',
    MeasurementTrackerEvent = 'MeasurementTrackerEvent',
    GeometricInnerState = False,
    AlgorithmName = 'gsf'
)


