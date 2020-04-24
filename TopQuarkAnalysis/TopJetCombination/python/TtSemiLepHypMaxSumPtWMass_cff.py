import FWCore.ParameterSet.Config as cms

#
# produce maxSumPtWMass hypothesis with all necessary 
# ingredients
#

## configure maxSumPtWMass hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMaxSumPtWMass_cfi import *
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMaxSumPtWMass_cfi import *

## make hypothesis
makeHypothesis_maxSumPtWMassTask = cms.Task(
    findTtSemiLepJetCombMaxSumPtWMass,
    ttSemiLepHypMaxSumPtWMass
)
makeHypothesis_maxSumPtWMass = cms.Sequence(makeHypothesis_maxSumPtWMassTask)
