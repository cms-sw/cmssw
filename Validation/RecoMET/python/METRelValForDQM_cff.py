import FWCore.ParameterSet.Config as cms
# File: METRelValForDQM.cff
# Author:  R. Remington
# Date: 03.01.09
# Fill validation histograms for MET.

from Validation.RecoMET.CaloMET_cfi import *
metAnalyzer.FineBinning = cms.untracked.bool(False)
metAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metHOAnalyzer.FineBinning = cms.untracked.bool(False)
metHOAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metOptAnalyzer.FineBinning = cms.untracked.bool(False)
metOptAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metOptHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptHOAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metOptNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

metOptNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

from Validation.RecoMET.PFMET_cfi import *
pfMetAnalyzer.FineBinning = cms.untracked.bool(False)
pfMetAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

from Validation.RecoMET.TCMET_cfi import *
tcMetAnalyzer.FineBinning = cms.untracked.bool(False)
tcMetAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

from Validation.RecoMET.MuonCorrectedCaloMET_cff import *
corMetGlobalMuonsAnalyzer.FineBinning = cms.untracked.bool(False)
corMetGlobalMuonsAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

from Validation.RecoMET.GenMET_cfi import *
genMetTrueAnalyzer.FineBinning = cms.untracked.bool(False)
genMetTrueAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

genMetCaloAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

genMetCaloAndNonPromptAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAndNonPromptAnalyzer.FolderName = cms.untracked.string("JetMET/MET/")

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
    corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer*
    genMetCaloAnalyzer*
    genMetCaloAndNonPromptAnalyzer)

    

