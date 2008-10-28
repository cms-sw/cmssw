import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(0.971213),
    MaxEta = cms.untracked.double(0.971213),
    MinPhi = cms.untracked.double(0.115112 ),
    MaxPhi = cms.untracked.double(0.115112 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
