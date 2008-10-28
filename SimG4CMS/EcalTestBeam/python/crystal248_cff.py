import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(0.221605),
    MaxEta = cms.untracked.double(0.221605),
    MinPhi = cms.untracked.double(0.0467487 ),
    MaxPhi = cms.untracked.double(0.0467487 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
