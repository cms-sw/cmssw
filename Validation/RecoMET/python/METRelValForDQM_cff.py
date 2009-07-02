import FWCore.ParameterSet.Config as cms
# File: METRelValForDQM.cff
# Author:  R. Remington
# Date: 03.01.09
# Fill validation histograms for MET.

from Validation.RecoMET.CaloMET_cfi import *
metAnalyzer.FineBinning = cms.untracked.bool(False)
metHOAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptAnalyzer.FineBinning = cms.untracked.bool(False)
metOptHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)

from Validation.RecoMET.PFMET_cfi import *
pfMetAnalyzer.FineBinning = cms.untracked.bool(False)

from Validation.RecoMET.TCMET_cfi import *
tcMetAnalyzer.FineBinning = cms.untracked.bool(False)

from Validation.RecoMET.GenMET_cfi import *
genMetTrueAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAndNonPromptAnalyzer.FineBinning = cms.untracked.bool(False)

METRelValSequence = cms.Sequence(
    metAnalyzer*
    metHOAnalyzer*
    metNoHFAnalyzer*
    metNoHFHOAnalyzer*
    metOptAnalyzer*
    metOptHOAnalyzer*
    metOptNoHFAnalyzer*
    metOptNoHFHOAnalyzer*
    pfMetAnalyzer*
    tcMetAnalyzer*
    genMetTrueAnalyzer*
    genMetCaloAnalyzer*
    genMetCaloAndNonPromptAnalyzer)

    

