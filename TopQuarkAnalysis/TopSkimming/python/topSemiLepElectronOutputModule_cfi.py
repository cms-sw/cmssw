import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topSemiLepElectron_AODSIMEventContent_cff import *
topSemiLepElectronOutputModule = cms.OutputModule("PoolOutputModule",
    topSemiLepElectronPlus1JetEventSelection,
    topSemiLepElectronAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topSemiLepElectron'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topSemiLepElectron.root')
)


