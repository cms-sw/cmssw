import FWCore.ParameterSet.Config as cms

#
# produce genmatch hypothesis with all necessary 
# ingredients
#

## std sequence to produce ttSemiJetPartonMatch
from TopQuarkAnalysis.TopTools.TtSemiLepJetPartonMatch_cfi import *

## configure genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGenMatch_cfi import *

## make hypothesis
makeHypothesis_genMatchTask = cms.Task(
    ttSemiLepJetPartonMatch,
    ttSemiLepHypGenMatch
)
makeHypothesis_genMatch = cms.Sequence(makeHypothesis_genMatchTask)
