import FWCore.ParameterSet.Config as cms

gemDigiHarvesting = cms.EDAnalyzer("MuonGEMDigisHarvestor",
  dbePath = cms.string("MuonGEMDigisV/GEMDigisTask/"),
  compareDBEPath = cms.string("MuonGEMHitsV/GEMHitsTask/"),
  dbeHistPrefix = cms.string("copad_dcEta"),
  compareDBEHistPrefix = cms.string("hit_dcEta"),
  detailPlot = cms.bool(False), 
)
MuonGEMDigisPostProcessors = cms.Sequence( gemDigiHarvesting ) 
