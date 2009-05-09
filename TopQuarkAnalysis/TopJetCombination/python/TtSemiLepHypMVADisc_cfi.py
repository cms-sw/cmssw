import FWCore.ParameterSet.Config as cms

#
# module to make the mvaDiscriminator hypothesis
#
ttSemiLepHypMVADisc = cms.EDProducer("TtSemiLepHypMVADisc",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input                                 
    jets  = cms.InputTag("selectedLayer1Jets"),
    ##lepton input                     
    leps  = cms.InputTag("selectedLayer1Muons"),
    ## mva input
    match = cms.InputTag("findTtSemiLepJetCombMVA")
)


