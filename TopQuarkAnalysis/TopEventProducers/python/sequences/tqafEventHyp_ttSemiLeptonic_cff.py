import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients
#

## considered event hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMaxSumPtWMass_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisGenMatch_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMVADisc_cff import *

## ttsemiEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiEventBuilder_cfi import *

## make all considered event hypotheses
makeTtSemiHyps  = cms.Sequence(makeHypothesis_maxSumPtWMass
                               *makeHypothesis_genMatch
                               *makeHypothesis_mvaDisc)

## fill ttSemiEvent structure
makeTtSemiEvent = cms.Sequence(makeTtSemiHyps*ttSemiEvent)

