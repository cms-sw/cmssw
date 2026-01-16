import FWCore.ParameterSet.Config as cms

from Validation.EcalClusters.egammaSuperClusterAnalyzer_cfi import *

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
for e in [peripheralPbPb, pp_on_AA, pp_on_XeXe_2017]:
    e.toModify(egammaSuperClusterAnalyzer, barrelCorSuperClusterCollection = cms.InputTag("correctedIslandBarrelSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, barrelRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, endcapCorSuperClusterCollection = cms.InputTag("correctedIslandEndcapSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, endcapRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters"))

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(egammaSuperClusterAnalyzer,
    enableEndcaps = False,
    barrelRecHitCollection = "ecalRecHit:EcalRecHitsEB",
    endcapRecHitCollection = None,
    endcapRawSuperClusterCollection = None,
    endcapCorSuperClusterCollection = None,
    endcapPreSuperClusterCollection = None,
    hist_bins_preshowerE = None,
    hist_min_preshowerE = None,
    hist_max_preshowerE = None
)
