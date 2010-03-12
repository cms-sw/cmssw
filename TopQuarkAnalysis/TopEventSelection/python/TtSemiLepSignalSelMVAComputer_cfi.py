import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepSignalSelMVA = cms.EDProducer("TtSemiLepSignalSelMVAComputer",
    ## met input
    mets  = cms.InputTag("patMETs"),
    ## jet input
    jets  = cms.InputTag("selectedPatJets"),
    ## lepton input
    muons = cms.InputTag("selectedPatMuons"),
    elecs = cms.InputTag("selectedPatElectrons")
)
