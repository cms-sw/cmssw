import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(0.658212),
    MaxEta = cms.double(0.658212),
    MinPhi = cms.double(-0.0400034 ),
    MaxPhi = cms.double(-0.0400034 ),
    Psi    = cms.double(999.9),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
