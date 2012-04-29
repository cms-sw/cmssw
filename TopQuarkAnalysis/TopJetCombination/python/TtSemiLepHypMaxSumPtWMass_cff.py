import FWCore.ParameterSet.Config as cms

#
# produce maxSumPtWMass hypothesis with all necessary 
# ingredients
#

## configure maxSumPtWMass hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMaxSumPtWMass_cfi import *

## make hypothesis
makeHypothesis_maxSumPtWMass = cms.Sequence(ttSemiLepHypMaxSumPtWMass)

