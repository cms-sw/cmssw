import FWCore.ParameterSet.Config as cms

me0DigiHarvesting = cms.EDAnalyzer("MuonME0DigisHarvestor")
MuonME0DigisPostProcessors = cms.Sequence( me0DigiHarvesting )

me0SegHarvesting = cms.EDAnalyzer("MuonME0SegHarvestor")
MuonME0SegPostProcessors = cms.Sequence( me0SegHarvesting )
