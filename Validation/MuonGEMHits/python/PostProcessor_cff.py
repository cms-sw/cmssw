import FWCore.ParameterSet.Config as cms



gemHitHarvesting = cms.EDAnalyzer("MuonGEMHits_Harvesting")
MuonGEMHitsPostProcessors = cms.Sequence( gemHitHarvesting ) 

