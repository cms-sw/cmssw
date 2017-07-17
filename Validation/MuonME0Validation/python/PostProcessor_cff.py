import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

me0DigiHarvesting = DQMEDHarvester("MuonME0DigisHarvestor")
MuonME0DigisPostProcessors = cms.Sequence( me0DigiHarvesting )

me0SegHarvesting = DQMEDHarvester("MuonME0SegHarvestor")
MuonME0SegPostProcessors = cms.Sequence( me0SegHarvesting )
