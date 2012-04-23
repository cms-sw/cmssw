import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topFullyHadronic_AODSIMEventContent_cff import *
topFullyHadronicOutputModule = cms.OutputModule("PoolOutputModule",
    topFullyHadronicAODSIMEventContent,
    topFullyHadronicEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topFullyHadronic'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topFullyHadronic.root')
)


