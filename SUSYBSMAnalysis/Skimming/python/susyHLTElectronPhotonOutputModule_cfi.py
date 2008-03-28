import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyElectronPhoton_HLT_EventContent_cff import *
susyHLTElectronPhotonOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMSusyHLTElectronPhotonEventContent,
    susyHLTElectronPhotonEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyHLTElectronPhoton'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyHLTElectronPhoton.root')
)


