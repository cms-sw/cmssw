import FWCore.ParameterSet.Config as cms

#
# module to make the mvaDiscriminator hypothesis
#
ttSemiLepHypMVADisc = cms.EDProducer("TtSemiLepHypMVADisc",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag("findTtSemiJetCombMVA")
)


