import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyJetMET_HLT_EventContent_cff import *
susyHLTJetMETOutputModule = cms.OutputModule("PoolOutputModule",
    susyHLTJetMETEventSelection,
    AODSIMSusyHLTJetMETEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyHLTJetMET'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyHLTJetMET.root')
)


