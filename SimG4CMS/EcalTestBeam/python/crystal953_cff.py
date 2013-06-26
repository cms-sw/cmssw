import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(0.834591),
    MaxEta = cms.double(0.834591),
    MinPhi = cms.double(-0.0402127 ),
    MaxPhi = cms.double(-0.0402127 ),
    Psi    = cms.double(999.9),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
