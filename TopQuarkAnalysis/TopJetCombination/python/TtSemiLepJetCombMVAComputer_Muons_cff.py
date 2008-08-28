import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVAComputer_Muons_cfi import *

## path for mva input file
TtSemiLepJetCombMVAFileSource = cms.ESSource("TtSemiLepJetCombMVAFileSource",
    ttSemiLepJetCombMVA = cms.FileInPath('TopQuarkAnalysis/TopJetCombination/data/TtSemiLepJetComb_Muons.mva')
)
