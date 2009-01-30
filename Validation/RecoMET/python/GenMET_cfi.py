import FWCore.ParameterSet.Config as cms

# File: GenMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
genMetAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_genMet.root'),
    InputMETLabel = cms.InputTag("genMet"),
    METType = cms.untracked.string('GenMET')
)

genMetNoNuBSMAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_genMetNoNuBSM.root'),
    InputMETLabel = cms.InputTag("genMetNoNuBSM"),
    METType = cms.untracked.string('GenMET')
)



genMetTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string("METTester_genMetTrue.root"),
    InputMETLabel = cms.InputTag("genMetTrue"),
    METType = cms.InputTag("GenMet")
    )

genMetCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string("METTester_genMetCalo.root"),
    InputMETLabel = cms.InputTag("genMetCalo"),
    METType = cms.InputTag("GenMet")
    )



genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string("METTester_genMetCaloAndNonPrompt.root"),
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt"),
    METType = cms.InputTag("GenMet")
    )




    
