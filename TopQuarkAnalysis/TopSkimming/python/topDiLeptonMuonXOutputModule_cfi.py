import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topDiLeptonMuonX_AODSIMEventContent_cff import *
topDiLeptonMuonXOutputModule = cms.OutputModule("PoolOutputModule",
    topDiLeptonMuonXEventSelection,
    topDiLeptonMuonXAODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('topDiLeptonMuonX'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('topDiLeptonMuonX.root')
)


