import FWCore.ParameterSet.Config as cms
genJetClosestMatchSelector = cms.EDFilter("GenJetClosestMatchSelector",
                              src = cms.InputTag("ak4GenJets"),
                              MatchTo = cms.InputTag("tauGenJetsSelectorAllHadrons")
)


# foo bar baz
# 6NV2MSfBxXMgP
# eCm2tN1Eoameu
