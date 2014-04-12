import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(0.971213),
    MaxEta = cms.double(0.971213),
    MinPhi = cms.double(0.115112 ),
    MaxPhi = cms.double(0.115112 ),
    Psi    = cms.double(999.9),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
