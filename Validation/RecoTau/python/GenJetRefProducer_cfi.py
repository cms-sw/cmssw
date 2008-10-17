import FWCore.ParameterSet.Config as cms

GenJetProducer  = cms.EDProducer("GenJetRefProducer",
                              GenJetSrc     = cms.untracked.InputTag("iterativeCone5GenJets"),
                              ptMinGenJet  = cms.untracked.double(5.0),
                              EtaMax         = cms.untracked.double(2.5)
)
