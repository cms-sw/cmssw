import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

cscDigiHarvesting = DQMEDHarvester("MuonCSCDigisHarvestor")
MuonCSCDigisPostProcessors = cms.Sequence(cscDigiHarvesting)
