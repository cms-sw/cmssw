import FWCore.ParameterSet.Config as cms
genJetClosestMatchSelector = cms.EDFilter("GenJetClosestMatchSelector",
                              src = cms.InputTag("ak4GenJetsNoNu"),
                              MatchTo = cms.InputTag("tauGenJetsSelectorAllHadrons")
)


