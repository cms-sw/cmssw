import FWCore.ParameterSet.Config as cms

me0DigiHarvesting = cms.EDAnalyzer("MuonME0DigisHarvestor")
MuonME0DigisPostProcessors = cms.Sequence( me0DigiHarvesting )
