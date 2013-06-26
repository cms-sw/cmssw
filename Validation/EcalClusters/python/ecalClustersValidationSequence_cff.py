import FWCore.ParameterSet.Config as cms

# analyzer to make histos from basic clusters
from Validation.EcalClusters.egammaBCAnalyzer_cfi import *
# analyzer to make histos from super clusters
from Validation.EcalClusters.egammaSCAnalyzer_cfi import *
ecalClustersValidationSequence = cms.Sequence(egammaBasicClusterAnalyzer+egammaSuperClusterAnalyzer)


