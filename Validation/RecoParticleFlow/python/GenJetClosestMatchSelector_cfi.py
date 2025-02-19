import FWCore.ParameterSet.Config as cms
genJetClosestMatchSelector = cms.EDFilter("GenJetClosestMatchSelector",
                              src = cms.InputTag("iterativeCone5GenJets"),
                              MatchTo = cms.InputTag("tauGenJetsSelectorAllHadrons")
)


