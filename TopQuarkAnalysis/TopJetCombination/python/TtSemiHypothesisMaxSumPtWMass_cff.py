import FWCore.ParameterSet.Config as cms

#
# produce maxSumPtWMass hypothesis with all necessary 
# ingredients
#
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMaxSumPtWMass_cfi import *

makeHypothesis_maxSumPtWMass = cms.Sequence(ttSemiHypothesisMaxSumPtWMass)

