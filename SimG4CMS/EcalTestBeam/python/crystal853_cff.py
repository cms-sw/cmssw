import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(0.74392),
    MaxEta = cms.untracked.double(0.74392),
    MinPhi = cms.untracked.double(-0.0401051 ),
    MaxPhi = cms.untracked.double(-0.0401051 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
