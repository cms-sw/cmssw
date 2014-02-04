import FWCore.ParameterSet.Config as cms

gemDigiHarvesting = cms.EDAnalyzer("MuonGEMDigis_Harvesting")
MuonGEMDigisPostProcessors = cms.Sequence(gemDigiHarvesting)
