import FWCore.ParameterSet.Config as cms

#
# make simple analysis plots for a comparison
# between event hypothesis from different algorithms
#

# initialize analyzers
from TopQuarkAnalysis.Examples.HypothesisAnalyzer_cfi import *
analyzeGenMatch      = analyzeHypothesis.clone()
analyzeMaxSumPtWMass = analyzeHypothesis.clone()
analyzeMVADisc       = analyzeHypothesis.clone()

# configure analyzers
analyzeGenMatch.hypoClassKey      = "kGenMatch"
analyzeMaxSumPtWMass.hypoClassKey = "kMaxSumPtWMass"
analyzeMVADisc.hypoClassKey       = "kMVADisc"

# define sequence
analyzeHypotheses = cms.Sequence(analyzeGenMatch *
                                 analyzeMaxSumPtWMass *
                                 analyzeMVADisc)
