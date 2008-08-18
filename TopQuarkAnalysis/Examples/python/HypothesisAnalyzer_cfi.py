import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of top event hypothese
#
analyzeHypothesis = cms.EDAnalyzer("HypothesisAnalyzer",
    hypoKey = cms.InputTag("ttSemiHypothesisMaxSumPtWMass","Key"),
    semiLepEvent = cms.InputTag("ttSemiLepEvent")
)


