import FWCore.ParameterSet.Config as cms

#
#  Author: Michael A. Balazs, University of Virginia
#  $Id: egammaSCAnalyzer.cfi,v 1.8 2006/12/13 15:03:22 mabalazs Exp $
#
from Validation.EcalClusters.VerificationCommonParameters_cfi import *
egammaSuperClusterAnalyzer = cms.EDFilter("EgammaSuperClusters",
    VerificationCommonParameters,
    hist_max_Size = cms.double(5.0),
    islandBarrelSuperClusterCollection = cms.InputTag("correctedIslandBarrelSuperClusters"),
    hist_min_EToverTruth = cms.double(0.5),
    hist_max_deltaEta = cms.double(0.025),
    hist_max_EToverTruth = cms.double(1.5),
    hist_min_Eta = cms.double(-2.5),
    hist_bins_S25toE = cms.int32(50),
    hist_min_deltaEta = cms.double(-0.025),
    islandEndcapSuperClusterCollection = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    hist_bins_Eta = cms.int32(91),
    hist_max_S25toE = cms.double(1.0),
    islandBarrelClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandBarrelShapeAssoc"),
    hist_bins_Phi = cms.int32(181),
    hybridBarrelSuperClusterCollection = cms.InputTag("correctedHybridSuperClusters"),
    hist_bins_EToverTruth = cms.int32(100),
    hist_bins_S1toS9 = cms.int32(50),
    hist_min_ET = cms.double(0.0),
    hist_bins_deltaEta = cms.int32(51),
    hist_max_S1toS9 = cms.double(1.0),
    hist_max_ET = cms.double(50.0),
    hybridBarrelClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
    hist_max_Eta = cms.double(2.5),
    hist_max_Phi = cms.double(3.14159),
    hist_min_Size = cms.double(0.0),
    islandEndcapClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc"),
    hist_min_S1toS9 = cms.double(0.0),
    hist_min_NumBC = cms.double(0.0),
    hist_bins_Size = cms.int32(5),
    hist_min_S25toE = cms.double(0.0),
    hist_min_Phi = cms.double(-3.14159),
    hist_max_NumBC = cms.double(10.0),
    hist_bins_ET = cms.int32(50),
    hist_bins_NumBC = cms.int32(10)
)



