import FWCore.ParameterSet.Config as cms

## import MVA computer cfi
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVAComputer_cfi import *

## path for mva input file
TtSemiLepJetCombMVAFileSource = cms.ESSource("TtSemiLepJetCombMVAFileSource",
    ttSemiLepJetCombMVA = cms.FileInPath('TopQuarkAnalysis/TopJetCombination/data/TtSemiLepJetComb.mva')
)
