import FWCore.ParameterSet.Config as cms

me0DigiHarvesting = cms.EDProducer("MuonME0DigisHarvestor")
MuonME0DigisPostProcessors = cms.Sequence( me0DigiHarvesting )

me0SegHarvesting = cms.EDProducer("MuonME0SegHarvestor")
MuonME0SegPostProcessors = cms.Sequence( me0SegHarvesting )
