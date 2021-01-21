import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemRecHitHarvesting = DQMEDHarvester("MuonGEMRecHitsHarvestor",
    regionIds = cms.untracked.vint32(-1, 1),
    stationIds = cms.untracked.vint32(1, 2),
    layerIds = cms.untracked.vint32(1, 2, 3, 4, 5, 6),
)


MuonGEMRecHitsPostProcessors = cms.Sequence( gemRecHitHarvesting ) 
