import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(0.658212),
    MaxEta = cms.untracked.double(0.658212),
    MinPhi = cms.untracked.double(-0.0400034 ),
    MaxPhi = cms.untracked.double(-0.0400034 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
