import FWCore.ParameterSet.Config as cms

BoostedTopProducer = cms.EDProducer("BoostedTopProducer",
    electronLabel = cms.InputTag("selectedLayer1Electrons"),
    muonLabel = cms.InputTag("selectedLayer1Muons"),
    jetLabel = cms.InputTag("selectedLayer1Jets"),
    caloIsoCut = cms.double(0.2),
    mTop = cms.double(175.0),
    solLabel = cms.InputTag("solutions"),
    metLabel = cms.InputTag("layer1METs")
)
