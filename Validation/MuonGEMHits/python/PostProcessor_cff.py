import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemSimHarvesting = DQMEDHarvester("MuonGEMHitsHarvestor")
MuonGEMHitsPostProcessors = cms.Sequence( gemSimHarvesting ) 
