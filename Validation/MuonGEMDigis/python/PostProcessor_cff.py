import FWCore.ParameterSet.Config as cms

gemDigiHarvesting = cms.EDAnalyzer("MuonGEMDigisHarvestor",
  dbePath = cms.string("MuonGEMDigisV/GEMDigisTask/"),
  compareDBEPath = cms.string("MuonGEMHitsV/GEMHitsTask/"),
  dbeHistPrefix = cms.string("copad_dcEta"),
#  dbeStripPrefix = cms.string("strip_dcEta_trk"),
#  dbePadPrefix = cms.string("pad_dcEta_trk"),
#  dbeCopadPrefix = cms.string("copad_dcEta_trk"),
  compareDBEHistPrefix = cms.string("hit_dcEta"),
  detailPlot = cms.bool(False), 
)
MuonGEMDigisPostProcessors = cms.Sequence( gemDigiHarvesting ) 
