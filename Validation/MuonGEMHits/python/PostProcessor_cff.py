import FWCore.ParameterSet.Config as cms

gemSimHarvesting = cms.EDAnalyzer("MuonGEMHitsHarvestor")
MuonGEMHitsPostProcessors = cms.Sequence( gemSimHarvesting ) 
