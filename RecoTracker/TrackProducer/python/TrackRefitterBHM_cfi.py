import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.TrackRefitter_cfi 
TrackRefitterBHM  = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src                 = "ctfWithMaterialTracksBeamHaloMuon",
    Fitter              = 'KFFittingSmootherBH',
    Propagator          = 'BeamHaloPropagatorAlong',
    GeometricInnerState = True
)
