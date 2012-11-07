import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.TrackRefitter_cfi 
TrackRefitterBHM  = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = cms.InputTag("ctfWithMaterialTracksBeamHaloMuon"),
    Fitter = cms.string('KFFittingSmootherBH'),
    Propagator = cms.string('BeamHaloPropagatorAlong'),
    GeometricInnerState = cms.bool(True)
)


