import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVAComputer_Muons_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMVADisc_cfi import *
makeHypothesis_mvaDisc = cms.Sequence(findTtSemiJetCombMVA*ttSemiHypothesisMVADisc)

