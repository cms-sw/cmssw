import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of top event hypotheses
#
analyzeHypothesis = cms.EDAnalyzer("HypothesisAnalyzer",
    semiLepEvent = cms.InputTag("ttSemiLepEvent"),
    hypoClassKey = cms.string("kMaxSumPtWMass")
)


