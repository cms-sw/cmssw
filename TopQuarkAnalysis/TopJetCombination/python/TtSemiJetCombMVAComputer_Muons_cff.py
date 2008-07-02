import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVAComputer_Muons_cfi import *
TtSemiJetCombMVAFileSource = cms.ESSource("TtSemiJetCombMVAFileSource",
    ttSemiJetCombMVA = cms.FileInPath('TtSemiJetComb_Muons.mva')
)


