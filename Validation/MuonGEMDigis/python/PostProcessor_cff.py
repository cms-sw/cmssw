import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemDigiHarvesting = DQMEDHarvester("MuonGEMDigisHarvestor",
    folder = cms.untracked.string("MuonGEMDigisV/GEMDigisTask/"),
    stripFolder = cms.untracked.string("MuonGEMDigisV/GEMDigisTask/Strip/"),
    padFolder = cms.untracked.string("MuonGEMDigisV/GEMDigisTask/Pad/"),
    copadFolder = cms.untracked.string("MuonGEMDigisV/GEMDigisTask/Copad/"),
    clusterFolder = cms.untracked.string("MuonGEMDigisV/GEMDigisTask/Cluster/"),
    logCategory=cms.untracked.string("MuonGEMDigisHarvestor"),
    regionIds = cms.untracked.vint32(-1, 1),
    stationIds = cms.untracked.vint32(1, 2),
    layerIds = cms.untracked.vint32(1, 2),
)
MuonGEMDigisPostProcessors = cms.Sequence( gemDigiHarvesting )
