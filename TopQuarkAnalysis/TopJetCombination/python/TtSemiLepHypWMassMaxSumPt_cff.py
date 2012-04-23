import FWCore.ParameterSet.Config as cms

#
# produce wMassMaxSumPt hypothesis with all necessary 
# ingredients
#

## configure wMassMaxSumPt hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassMaxSumPt_cfi import *

## make hypothesis
makeHypothesis_wMassMaxSumPt = cms.Sequence(ttSemiLepHypWMassMaxSumPt)
