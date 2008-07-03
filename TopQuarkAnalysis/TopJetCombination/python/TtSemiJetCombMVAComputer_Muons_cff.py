import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVAComputer_Muons_cfi import *

## path for mva input file
TtSemiJetCombMVAFileSource = cms.ESSource("TtSemiJetCombMVAFileSource",
    ttSemiJetCombMVA = cms.FileInPath('TopQuarkAnalysis/TopJetCombination/data/TtSemiJetComb_Muons.mva')
)
