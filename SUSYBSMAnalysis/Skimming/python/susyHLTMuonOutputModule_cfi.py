import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyMuon_HLT_EventContent_cff import *
susyHLTMuonOutputModule = cms.OutputModule("PoolOutputModule",
    susyHLTMuonEventSelection,
    AODSIMSusyHLTMuonEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyHLTMuon'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyHLTMuon.root')
)


