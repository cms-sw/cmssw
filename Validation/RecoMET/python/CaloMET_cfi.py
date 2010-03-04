import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET.
metAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_met.root'),
    InputMETLabel = cms.InputTag("met"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metHO.root'),
    InputMETLabel = cms.InputTag("metHO"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metNoHF.root'),
    InputMETLabel = cms.InputTag("metNoHF"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")

    )

metNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metNoHFHO.root'),
    InputMETLabel = cms.InputTag("metNoHFHO"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metOptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metOpt.root'),
    InputMETLabel = cms.InputTag("metOpt"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")

    )

metOptHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metOptHO.root'),
    InputMETLabel = cms.InputTag("metOptHO"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metOptNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metOptNoHF.root'),
    InputMETLabel = cms.InputTag("metOptNoHF"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    
    )

metOptNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester_metOptNoHFHO.root'),
    InputMETLabel = cms.InputTag("metOptNoHFHO"),
    METType = cms.untracked.string('CaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")

    )



