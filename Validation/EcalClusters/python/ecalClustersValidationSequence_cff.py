import FWCore.ParameterSet.Config as cms

# analyzer to make histos from basic clusters
from Validation.EcalClusters.egammaBasicClusterAnalyzer_cff import *
# analyzer to make histos from super clusters
from Validation.EcalClusters.egammaSuperClusterAnalyzer_cff import *
ecalClustersValidationSequence = cms.Sequence(egammaBasicClusterAnalyzer+egammaSuperClusterAnalyzer)


