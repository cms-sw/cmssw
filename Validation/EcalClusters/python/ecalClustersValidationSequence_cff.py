import FWCore.ParameterSet.Config as cms

# analyzer to make histos from basic clusters
from Validation.EcalClusters.egammaBCAnalyzer_cfi import *
# analyzer to make histos from super clusters
from Validation.EcalClusters.egammaSCAnalyzer_cfi import *
VerificationCommonParameters = cms.PSet(
    CMSSW_Version = cms.untracked.string('V2_0_0_pre6'),
    MCTruthCollection = cms.InputTag("source"),
    outputFile = cms.untracked.string('EcalClustersValidation.root'),
    verboseDBE = cms.untracked.bool(True)
)
ecalClustersValidationSequence = cms.Sequence(egammaBasicClusterAnalyzer+egammaSuperClusterAnalyzer)

