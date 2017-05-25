import FWCore.ParameterSet.Config as cms

gemSimHarvesting = cms.EDProducer("MuonGEMHitsHarvestor")
MuonGEMHitsPostProcessors = cms.Sequence( gemSimHarvesting ) 
