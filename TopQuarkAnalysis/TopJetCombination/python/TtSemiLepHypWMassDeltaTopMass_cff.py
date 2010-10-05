import FWCore.ParameterSet.Config as cms

#
# produce wMassDeltaTopMass hypothesis with all necessary 
# ingredients
#

## configure wMassDeltaTopMass hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassDeltaTopMass_cfi import *

## make hypothesis
makeHypothesis_wMassDeltaTopMass = cms.Sequence(ttSemiLepHypWMassDeltaTopMass)
