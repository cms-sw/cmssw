import FWCore.ParameterSet.Config as cms

#
# module to make the mvaDiscriminator hypothesis
#
ttSemiHypothesisMVADisc = cms.EDProducer("TtSemiHypothesisMVADisc",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag("findTtSemiJetCombMVA")
)


