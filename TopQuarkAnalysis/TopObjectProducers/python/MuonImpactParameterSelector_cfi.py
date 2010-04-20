import FWCore.ParameterSet.Config as cms

impactParameterMuons = cms.EDProducer("MuonImpactParameterSelector",
    vertices = cms.InputTag("offlinePrimaryVertices"),
    leptons  = cms.InputTag("selectedPatMuons"),
    cut      = cms.double(3)
)
