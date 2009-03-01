import FWCore.ParameterSet.Config as cms

# File: HTMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
htMetSC5Analyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("htMetSC5"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(True)
)


htMetSC7Analyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("htMetSC7"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(True)
)

htMetIC5Analyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("htMetIC5"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(True)
)

htMetKT4Analyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("htMetKT4"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(True)
)

htMetKT6Analyzer = cms.EDFilter(
    "METTester",
    InputMETLabel = cms.InputTag("htMetKT6"),
    METType = cms.untracked.string('MET'),
    FineBinning = cms.untracked.bool(True)
)


