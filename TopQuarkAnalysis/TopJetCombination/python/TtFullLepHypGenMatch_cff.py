import FWCore.ParameterSet.Config as cms

#
# produce genmatch hypothesis with all necessary 
# ingredients
#

## std sequence to produce ttFullJetPartonMatch
from TopQuarkAnalysis.TopTools.TtFullLepJetPartonMatch_cfi import *

## configure genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullLepHypGenMatch_cfi import *

## make hypothesis
makeHypothesis_genMatch = cms.Sequence(ttFullLepJetPartonMatch *
                                       ttFullLepHypGenMatch)

