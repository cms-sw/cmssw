import FWCore.ParameterSet.Config as cms

# File: GenMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
genMetAnalyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("genMet"),
    METType = cms.untracked.string('GenMET'),
    FineBinning = cms.untracked.bool(True),                            
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
)

genMetNoNuBSMAnalyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("genMetNoNuBSM"),
    METType = cms.untracked.string('GenMET'),
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


genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt"),
    METType = cms.untracked.string("GenMET"),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )




    
