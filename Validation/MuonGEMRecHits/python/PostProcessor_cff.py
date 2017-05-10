import FWCore.ParameterSet.Config as cms

gemRecHitHarvesting = cms.EDAnalyzer("MuonGEMRecHitsHarvestor")
MuonGEMRecHitsPostProcessors = cms.Sequence( gemRecHitHarvesting ) 
