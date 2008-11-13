import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of top event hypothese
#
analyzeHypothesis = cms.EDAnalyzer("HypothesisAnalyzer",
    semiLepEvent = cms.InputTag("ttSemiLepEvent"),
    hypoKey      = cms.InputTag("ttSemiLepHypMaxSumPtWMass","Key")
)


