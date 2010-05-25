import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttFullLepHypGenMatch = cms.EDProducer("TtFullLepHypGenMatch",
    electrons = cms.InputTag("selectedPatElectrons"),
    muons     = cms.InputTag("selectedPatMuons"),
    jets      = cms.InputTag("selectedPatJets"),    
    mets      = cms.InputTag("patMETs"),
    match     = cms.InputTag("ttFullLepJetPartonMatch"), 
    jetCorrectionLevel = cms.string("abs")   
)


