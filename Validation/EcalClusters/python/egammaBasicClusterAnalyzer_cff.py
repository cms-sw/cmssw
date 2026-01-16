import FWCore.ParameterSet.Config as cms

from Validation.EcalClusters.egammaBasicClusterAnalyzer_cfi import *
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(egammaBasicClusterAnalyzer, enableEndcaps = False, endcapBasicClusterCollection = None)
