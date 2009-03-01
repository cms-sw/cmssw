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
    FineBinning = cms.untracked.bool(True)                            
)

genMetNoNuBSMAnalyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("genMetNoNuBSM"),
    METType = cms.untracked.string('GenMET'),
    FineBinning = cms.untracked.bool(True)                                                              
)



genMetTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetTrue"),
    METType = cms.InputTag("GenMet"),
    FineBinning = cms.untracked.bool(True)                             
    )

genMetCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetCalo"),
    METType = cms.InputTag("GenMet"),
    FineBinning = cms.untracked.bool(True)                             
    )


genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt"),
    METType = cms.InputTag("GenMet"),
    FineBinning = cms.untracked.bool(True)                             
    )




    
