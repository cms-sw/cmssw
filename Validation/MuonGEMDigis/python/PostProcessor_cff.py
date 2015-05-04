import FWCore.ParameterSet.Config as cms

gemDigiHarvesting = cms.EDAnalyzer("MuonGEMDigisHarvestor")
MuonGEMDigisPostProcessors = cms.Sequence( gemDigiHarvesting ) 
