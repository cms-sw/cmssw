import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(1.43963),
    MaxEta = cms.untracked.double(1.43963),
    MinPhi = cms.untracked.double(-0.0409942 ),
    MaxPhi = cms.untracked.double(-0.0409942 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
