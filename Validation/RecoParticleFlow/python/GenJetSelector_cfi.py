import FWCore.ParameterSet.Config as cms
genJetSelector = cms.EDFilter("GenJetSelector",
                              src = cms.InputTag("iterativeCone5GenJets"),
                              MatchTo = cms.InputTag("tauGenJets")
)


