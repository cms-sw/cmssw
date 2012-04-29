import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topLeptonTau_AODSIMEventContent_cff import *
topLeptonTauETauOutputModule = cms.OutputModule("PoolOutputModule",
    topLeptonTauETauEventSelection,
    topLeptonTauAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topLeptonTauETau'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topLeptonTauETau.root')
)

topLeptonTauMuTauOutputModule = cms.OutputModule("PoolOutputModule",
    topLeptonTauMuTauEventSelection,
    topLeptonTauAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topLeptonTauMuTau'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topLeptonTauMuTau.root')
)

topLeptonTauOutputModule = cms.OutputModule("PoolOutputModule",
    topLeptonTauEventSelection,
    topLeptonTauAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topLeptonTau'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topLeptonTau.root')
)


