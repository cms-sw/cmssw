import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(1.00607),
    MaxEta = cms.untracked.double(1.00607),
    MinPhi = cms.untracked.double(-0.0404424 ),
    MaxPhi = cms.untracked.double(-0.0404424 ),
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamPosition = cms.untracked.double(-26733.5)
)
