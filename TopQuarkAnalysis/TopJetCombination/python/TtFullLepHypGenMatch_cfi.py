import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttFullLepHypGenMatch = cms.EDProducer("TtFullLepHypGenMatch",
    electrons = cms.InputTag("selectedLayer1Electrons"),
    muons     = cms.InputTag("selectedLayer1Muons"),
    jets      = cms.InputTag("selectedLayer1Jets"),    
    mets      = cms.InputTag("layer1METs"),
    match     = cms.InputTag("ttFullLepJetPartonMatch")
)


