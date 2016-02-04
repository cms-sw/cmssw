import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(2.06),
    MaxEta = cms.double(2.06),
    MinPhi = cms.double(0.46 ),
    MaxPhi = cms.double(0.46 ),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
