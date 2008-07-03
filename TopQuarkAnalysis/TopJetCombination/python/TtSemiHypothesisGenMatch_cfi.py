import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttSemiHypothesisGenMatch = cms.EDProducer("TtSemiHypothesisGenMatch",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag("ttSemiJetPartonMatch")
)


