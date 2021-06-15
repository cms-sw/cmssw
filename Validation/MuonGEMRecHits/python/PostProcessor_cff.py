import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from Validation.MuonGEMHits.MuonGEMCommonParameters_cfi import GEMValidationCommonParameters

gemRecHitHarvesting = DQMEDHarvester("MuonGEMRecHitsHarvestor",
    GEMValidationCommonParameters,
    regionIds = cms.untracked.vint32(-1, 1),
    stationIds = cms.untracked.vint32(1),
    layerIds = cms.untracked.vint32(1, 2, 3, 4, 5, 6),
)

MuonGEMRecHitsPostProcessors = cms.Sequence( gemRecHitHarvesting )

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( gemRecHitHarvesting, stationIds = (1, 2) )
