import FWCore.ParameterSet.Config as cms
# File: METRelValForDQM.cff
# Author:  R. Remington
# Date: 03.01.09
# Fill validation histograms for MET.

from Validation.RecoMET.CaloMET_cfi import *
metAnalyzer.FineBinning = cms.untracked.bool(False)
metAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metHOAnalyzer.FineBinning = cms.untracked.bool(False)
metHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)
metNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metOptAnalyzer.FineBinning = cms.untracked.bool(False)
metOptAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metOptHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metOptNoHFAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

metOptNoHFHOAnalyzer.FineBinning = cms.untracked.bool(False)
metOptNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

from Validation.RecoMET.PFMET_cfi import *
pfMetAnalyzer.FineBinning = cms.untracked.bool(False)
pfMetAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

from Validation.RecoMET.TCMET_cfi import *
tcMetAnalyzer.FineBinning = cms.untracked.bool(False)
tcMetAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

from Validation.RecoMET.MuonCorrectedCaloMET_cff import *
corMetGlobalMuonsAnalyzer.FineBinning = cms.untracked.bool(False)
corMetGlobalMuonsAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

from Validation.RecoMET.GenMET_cfi import *
genMetTrueAnalyzer.FineBinning = cms.untracked.bool(False)
genMetTrueAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

genMetCaloAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

genMetCaloAndNonPromptAnalyzer.FineBinning = cms.untracked.bool(False)
genMetCaloAndNonPromptAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

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

    
#Removed the MET collections that we no longer monitor
#in an attempt to reduce the number of histograms produced
# as requested by DQM group to reduce the load on server. 
# -Samantha Hewamanage (samantha@cern.ch) - 04-27-2012

METValidation = cms.Sequence(
    metAnalyzer*
    #metHOAnalyzer*
    #metNoHFAnalyzer*
    #metNoHFHOAnalyzer*
    #metOptAnalyzer*
    #metOptHOAnalyzer*
    #metOptNoHFAnalyzer*
    #metOptNoHFHOAnalyzer*
    pfMetAnalyzer*
    tcMetAnalyzer*
    #corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer #*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer)

    


