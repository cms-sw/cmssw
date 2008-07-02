import FWCore.ParameterSet.Config as cms

#
# produce genMatch hypothesis with all necessary 
# ingredients
#
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVAComputer_Muons_cff import *
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMVADisc_cfi import *

makeHypothesis_mvaDisc = cms.Sequence(findTtSemiJetCombMVA*ttSemiHypothesisMVADisc)

