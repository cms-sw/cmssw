import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttSemiLepHypGenMatch = cms.EDProducer("TtSemiLepHypGenMatch",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input
    leps  = cms.InputTag("selectedLayer1Muons"),
    ## gen match hypothesis input
    match = cms.InputTag("ttSemiLepJetPartonMatch")
)


