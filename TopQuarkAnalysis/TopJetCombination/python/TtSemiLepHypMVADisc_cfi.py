import FWCore.ParameterSet.Config as cms

#
# module to make the mvaDiscriminator hypothesis
#
ttSemiLepHypMVADisc = cms.EDProducer("TtSemiLepHypMVADisc",
    ## met input
    mets  = cms.InputTag("patMETs"),
    ## jet input                                 
    jets  = cms.InputTag("selectedPatJets"),
    ##lepton input                     
    leps  = cms.InputTag("selectedPatMuons"),
    ## mva input
    match = cms.InputTag("findTtSemiLepJetCombMVA")
)


