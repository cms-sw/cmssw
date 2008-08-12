import FWCore.ParameterSet.Config as cms

#
# produce wMassMaxSumPt hypothesis with all necessary 
# ingredients
#

## configure wMassMaxSumPt hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepWMassMaxSumPt_cfi import *

## make hypothesis
makeHypothesis_wMassMaxSumPt = cms.Sequence(ttSemiLepWMassMaxSumPt)
