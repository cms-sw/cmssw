import FWCore.ParameterSet.Config as cms
                                        
common_beam_direction_parameters = cms.PSet(
    MinEta = cms.double(1.35393),
    MaxEta = cms.double(1.35393),
    MinPhi = cms.double(-0.040891 ),
    MaxPhi = cms.double(-0.040891 ),
    Psi    = cms.double(999.9),
    BeamMeanX = cms.double(0.0),
    BeamMeanY = cms.double(0.0),
    BeamPosition = cms.double(-26733.5)
)
