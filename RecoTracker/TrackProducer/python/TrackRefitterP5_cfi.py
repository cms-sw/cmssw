import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.TrackRefitter_cfi 
TrackRefitterP5  = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src                 = "ctfWithMaterialTracksP5",
    Fitter              = 'FittingSmootherRKP5',
    AlgorithmName       = 'ctf',
    GeometricInnerState = True
)
