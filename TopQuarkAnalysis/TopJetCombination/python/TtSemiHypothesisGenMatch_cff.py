import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopTools.TtSemiEvtJetPartonMatch_cfi import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisGenMatch_cfi import *
makeHypothesis_genMatch = cms.Sequence(ttSemiJetPartonMatch*ttSemiHypothesisGenMatch)

