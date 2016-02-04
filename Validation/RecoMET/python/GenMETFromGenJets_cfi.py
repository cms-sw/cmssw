import FWCore.ParameterSet.Config as cms

# File: GenMETFromGenJets.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
genMetIC5GenJetsAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("genMetIC5GenJets"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )



