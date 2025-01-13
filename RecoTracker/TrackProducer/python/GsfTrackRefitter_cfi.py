import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.gsfTrackRefitter_cfi import gsfTrackRefitter
GsfTrackRefitter = gsfTrackRefitter.clone(
    src = "pixelMatchGsfFit",
    beamSpot = "offlineBeamSpot",
    Fitter = "GsfElectronFittingSmoother",
    useHitsSplitting = False,
    TrajectoryInEvent = False,
    TTRHBuilder = "WithTrackAngle",
    Propagator = "fwdGsfElectronPropagator",
    constraint = '',
    #set to "", the secondary hit pattern will not be filled (backward compatible with DetLayer=0)
    NavigationSchool = '',
    MeasurementTracker = '',
    MeasurementTrackerEvent = 'MeasurementTrackerEvent',
    AlgorithmName = 'gsf'
)


