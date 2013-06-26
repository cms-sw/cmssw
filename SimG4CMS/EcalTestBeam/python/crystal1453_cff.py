import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(1.26821),
    MaxEta = cms.double(1.26821),
    MinPhi = cms.double(-0.040783 ),
    MaxPhi = cms.double(-0.040783 ),
    Psi    = cms.double(999.9),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
