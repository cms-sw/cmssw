import FWCore.ParameterSet.Config as cms

# File: GenMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.

genMptTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMptTrue"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMetTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetTrue"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMetCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetCalo"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMptCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMptCalo"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )


genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )




    
