import FWCore.ParameterSet.Config as cms

#
# make simple analysis plots for a comparison
# between a simple algorithmic and a gen match
# event hypothesis
#

# initialize analyzers
import TopQuarkAnalysis.Examples.HypothesisAnalyzer_cfi
analyzeMaxSumPtWMass = TopQuarkAnalysis.Examples.HypothesisAnalyzer_cfi.analyzeHypothesis.clone()
analyzeGenMatch      = TopQuarkAnalysis.Examples.HypothesisAnalyzer_cfi.analyzeHypothesis.clone()

# configure analyzers
analyzeMaxSumPtWMass.hypoClassKey = 'ttSemiLepHypMaxSumPtWMass:Key'
analyzeGenMatch.hypoClassKey      = 'ttSemiLepHypGenMatch:Key'

# define sequence
analyzeAllHypotheses = cms.Sequence(analyzeMaxSumPtWMass*analyzeGenMatch)
