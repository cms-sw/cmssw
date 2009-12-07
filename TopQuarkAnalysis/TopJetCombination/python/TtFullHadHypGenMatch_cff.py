import FWCore.ParameterSet.Config as cms

#
# produce genmatch hypothesis with all necessary 
# ingredients
#

## std sequence to produce ttFullHadJetPartonMatch
from TopQuarkAnalysis.TopTools.TtFullHadJetPartonMatch_cfi import *

## configure genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullHadHypGenMatch_cfi import *

## make hypothesis
makeHypothesis_genMatch = cms.Sequence(ttFullHadJetPartonMatch *
                                       ttFullHadHypGenMatch)

