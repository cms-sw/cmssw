import FWCore.ParameterSet.Config as cms

#
#  Author: Michael A. Balazs, University of Virginia
#  $Id: egammaBCAnalyzer_cfi.py,v 1.3 2008/05/28 20:07:58 fabiocos Exp $
#
from Validation.EcalClusters.VerificationCommonParameters_cfi import *
egammaBasicClusterAnalyzer = cms.EDFilter("EgammaBasicClusters",
    VerificationCommonParameters,
    hist_max_Size = cms.double(20.0),
    hist_min_Size = cms.double(0.0),
    hist_min_NumRecHits = cms.double(0.0),
    hist_bins_Eta = cms.int32(91),
    endcapBasicClusterCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    barrelBasicClusterCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    hist_bins_Size = cms.int32(20),
    hist_bins_Phi = cms.int32(181),
    hist_min_ET = cms.double(0.0),
    hist_max_NumRecHits = cms.double(50.0),
    hist_max_ET = cms.double(200.0),
    hist_min_Phi = cms.double(-3.14159),
    hist_min_Eta = cms.double(-2.5),
    hist_bins_NumRecHits = cms.int32(50),
    hist_max_Eta = cms.double(2.5),
    hist_max_Phi = cms.double(3.14159),
    hist_bins_ET = cms.int32(200)
)



