import FWCore.ParameterSet.Config as cms

# File: HTMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
htMetSC5Analyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_htMetSC5.root'),
    InputMETLabel = cms.InputTag("htMetSC5"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
)


htMetSC7Analyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_htMetSC7.root'),
    InputMETLabel = cms.InputTag("htMetSC7"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
)

htMetIC5Analyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_htMetIC5.root'),
    InputMETLabel = cms.InputTag("htMetIC5"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
)

htMetKT4Analyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_htMetKT4.root'),
    InputMETLabel = cms.InputTag("htMetKT4"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
)

htMetKT6Analyzer = cms.EDFilter(
    "METTester",
    OutputFile = cms.untracked.string('METTester_htMetKT6.root'),
    InputMETLabel = cms.InputTag("htMetKT6"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(False)
)


