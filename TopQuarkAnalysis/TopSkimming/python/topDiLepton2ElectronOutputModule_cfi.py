import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topDiLepton2Electron_AODSIMEventContent_cff import *
topDiLepton2ElectronOutputModule = cms.OutputModule("PoolOutputModule",
    topDiLepton2ElectronEventSelection,
    topDiLepton2ElectronAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topDiLepton2Electron'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topDiLepton2Electron.root')
)


