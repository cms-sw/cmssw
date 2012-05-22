import FWCore.ParameterSet.Config as cms

#
# make simple analysis plots for a comparison
# between event hypothesis from different algorithms
#

# initialize/configure analyzers
from TopQuarkAnalysis.Examples.HypothesisAnalyzer_cfi import *
analyzeGenMatch      = analyzeHypothesis.clone(hypoClassKey = "kGenMatch")
analyzeMaxSumPtWMass = analyzeHypothesis.clone(hypoClassKey = "kMaxSumPtWMass")
analyzeKinFit        = analyzeHypothesis.clone(hypoClassKey = "kKinFit")

# define sequence
analyzeHypotheses = cms.Sequence(analyzeGenMatch *
                                 analyzeMaxSumPtWMass *
                                 analyzeKinFit)
