import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.TrackRefitter_cfi 
TrackRefitterP5  = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = cms.InputTag("ctfWithMaterialTracksP5"),
    Fitter = cms.string('FittingSmootherRKP5'),
    #TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('ctf'),
    GeometricInnerState = cms.bool(True)
)


