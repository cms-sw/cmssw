import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttSemiLepHypGenMatch = cms.EDProducer("TtSemiLepHypGenMatch",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag("ttSemiLepJetPartonMatch")
)


