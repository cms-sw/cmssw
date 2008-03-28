import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyJetMET_EventContent_cff import *
susyJetMETOutputModule = cms.OutputModule("PoolOutputModule",
    susyJetMETEventSelection,
    AODSIMSusyJetMETEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyJetMET'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyJetMET.root')
)


