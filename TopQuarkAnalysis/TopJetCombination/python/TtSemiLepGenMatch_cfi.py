import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttSemiLepGenMatch = cms.EDProducer("TtSemiLepGenMatch",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag("ttSemiJetPartonMatch")
)


