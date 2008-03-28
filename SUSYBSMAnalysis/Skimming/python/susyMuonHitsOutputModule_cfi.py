import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyMuonHits_EventContent_cff import *
susyMuonHitsOutputModule = cms.OutputModule("PoolOutputModule",
    susyMuonHitsEventSelection,
    AODSIMSusyMuonHitsEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyMuonHits'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyMuonHits.root')
)


