import FWCore.ParameterSet.Config as cms

common_beam_direction_parameters = cms.PSet(
    BeamMeanY = cms.untracked.double(0.0),
    BeamMeanX = cms.untracked.double(0.0),
    MaxEta = cms.untracked.double(0.572519),
    MaxPhi = cms.untracked.double(-0.0398542),
    MinEta = cms.untracked.double(0.572519),
    BeamPosition = cms.untracked.double(-26733.5),
    MinPhi = cms.untracked.double(-0.0398542)
)

