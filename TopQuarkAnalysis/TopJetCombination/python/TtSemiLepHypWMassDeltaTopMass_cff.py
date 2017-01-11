import FWCore.ParameterSet.Config as cms

#
# produce wMassDeltaTopMass hypothesis with all necessary 
# ingredients
#

## configure wMassDeltaTopMass hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombWMassDeltaTopMass_cfi import *
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassDeltaTopMass_cfi import *

## make hypothesis
makeHypothesis_wMassDeltaTopMassTask = cms.Task(
    findTtSemiLepJetCombWMassDeltaTopMass,
    ttSemiLepHypWMassDeltaTopMass
)
makeHypothesis_wMassDeltaTopMass = cms.Sequence(makeHypothesis_wMassDeltaTopMassTask)
