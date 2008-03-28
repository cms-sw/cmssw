import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyMuon_EventContent_cff import *
susyMuonOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMSusyMuonEventContent,
    susyMuonEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyMuon'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyMuon.root')
)


