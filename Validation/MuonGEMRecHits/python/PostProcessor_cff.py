import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemRecHitHarvesting = DQMEDHarvester("MuonGEMRecHitsHarvestor")
MuonGEMRecHitsPostProcessors = cms.Sequence( gemRecHitHarvesting ) 
