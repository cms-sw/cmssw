import FWCore.ParameterSet.Config as cms

common_beam_direction_parameters = cms.PSet(
    MinE         = cms.double(50.0),
    MaxE         = cms.double(50.0),
    PartID       = cms.vint32(-211),
    MinEta       = cms.double(0.2175),
    MaxEta       = cms.double(0.2175),
    MinPhi       = cms.double(-0.1309),
    MaxPhi       = cms.double(-0.1309),
    BeamPosition = cms.double(-800.0)
    )
