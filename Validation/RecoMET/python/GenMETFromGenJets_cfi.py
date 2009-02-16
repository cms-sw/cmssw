import FWCore.ParameterSet.Config as cms

# File: GenMETFromGenJets.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
genMetIC5GenJetsAnalyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_genMetIC5GenJets.root'),
    InputMETLabel = cms.InputTag("genMetIC5GenJets"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
    )



