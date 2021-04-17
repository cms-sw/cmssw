import FWCore.ParameterSet.Config as cms

from Validation.EcalClusters.VerificationCommonParameters_cfi import *
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
egammaSuperClusterAnalyzer = DQMEDAnalyzer('EgammaSuperClusters',
    VerificationCommonParameters,
    barrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    barrelRawSuperClusterCollection = cms.InputTag("hybridSuperClusters"),
    barrelCorSuperClusterCollection = cms.InputTag("correctedHybridSuperClusters"),
    endcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    endcapRawSuperClusterCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    endcapCorSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    endcapPreSuperClusterCollection = cms.InputTag("multi5x5SuperClustersWithPreshower"),
    hist_max_Size = cms.double(5.0),
    hist_min_Size = cms.double(0.0),
    hist_bins_Size = cms.int32(5),                                          
    hist_bins_deltaR = cms.int32(50),
    hist_min_deltaR = cms.double(0.0),
    hist_max_deltaR = cms.double(0.5),
    hist_min_preshowerE = cms.double(0.0),                                          
    hist_bins_preshowerE = cms.int32(100),
    hist_max_preshowerE = cms.double(100.0),
    hist_bins_etaWidth = cms.int32(100),
    hist_min_etaWidth = cms.double(0.0),
    hist_max_etaWidth = cms.double(0.1),
    hist_bins_Eta = cms.int32(100),
    hist_min_Eta = cms.double(-3.0),
    hist_max_Eta = cms.double(3.0),
    hist_bins_EoverTruth = cms.int32(100),
    hist_min_EoverTruth = cms.double(0.5),
    hist_max_EoverTruth = cms.double(1.5),
    hist_bins_S25toE = cms.int32(50),
    hist_min_S25toE = cms.double(0.0),
    hist_max_S25toE = cms.double(1.0),
    hist_bins_phiWidth = cms.int32(100),
    hist_min_phiWidth = cms.double(0.0),
    hist_max_phiWidth = cms.double(0.1),
    hist_bins_Phi = cms.int32(181),
    hist_min_Phi = cms.double(-3.14159),
    hist_max_Phi = cms.double(3.14159),
    hist_bins_S1toS9 = cms.int32(50),
    hist_min_S1toS9 = cms.double(0.0),
    hist_max_S1toS9 = cms.double(1.0),
    hist_bins_ET = cms.int32(100),                                          
    hist_min_ET = cms.double(0.0),
    hist_max_ET = cms.double(50.0),
    hist_bins_NumBC = cms.int32(10),
    hist_min_NumBC = cms.double(0.0),
    hist_max_NumBC = cms.double(10.0),
    hist_bins_R = cms.int32(55),
    hist_min_R = cms.double(0.0),
    hist_max_R = cms.double(175.0)    

                                          
)


from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
for e in [peripheralPbPb, pp_on_AA, pp_on_XeXe_2017]:
    e.toModify(egammaSuperClusterAnalyzer, barrelCorSuperClusterCollection = cms.InputTag("correctedIslandBarrelSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, barrelRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, endcapCorSuperClusterCollection = cms.InputTag("correctedIslandEndcapSuperClusters"))
    e.toModify(egammaSuperClusterAnalyzer, endcapRawSuperClusterCollection = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters"))
