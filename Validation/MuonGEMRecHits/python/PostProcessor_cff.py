import FWCore.ParameterSet.Config as cms

gemRecHitHarvesting = cms.EDProducer("MuonGEMRecHitsHarvestor")
MuonGEMRecHitsPostProcessors = cms.Sequence( gemRecHitHarvesting ) 
