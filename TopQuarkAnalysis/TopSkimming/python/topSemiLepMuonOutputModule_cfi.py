import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topSemiLepMuon_AODSIMEventContent_cff import *
topSemiLepMuonOutputModule = cms.OutputModule("PoolOutputModule",
    topSemiLepMuonAODSIMEventContent,
    topSemiLepMuonPlus1JetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topSemiLepMuon'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topSemiLepMuon.root')
)


