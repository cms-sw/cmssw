import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyElectronPhoton_EventContent_cff import *
susyElectronPhotonOutputModule = cms.OutputModule("PoolOutputModule",
    susyElectronPhotonEventSelection,
    AODSIMSusyElectronPhotonEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('susyElectronPhoton'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('susyElectronPhoton.root')
)


