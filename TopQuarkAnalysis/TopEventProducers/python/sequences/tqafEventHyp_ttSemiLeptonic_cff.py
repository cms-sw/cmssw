import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMaxSumPtWMass_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisGenMatch_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMVADisc_cff import *
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiEventBuilder_cfi import *
makeTtSemiHyps = cms.Sequence(makeHypothesis_maxSumPtWMass*makeHypothesis_genMatch*makeHypothesis_mvaDisc)
makeTtSemiEvent = cms.Sequence(makeTtSemiHyps*ttSemiEvent)

