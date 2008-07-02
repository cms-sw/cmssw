import FWCore.ParameterSet.Config as cms

process = cms.Process("egammaAnalysis")
process.load("DQMServices.Core.DQM_cfg")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Validation.EcalClusters.egammaBCAnalyzer_cfi")
process.load("Validation.EcalClusters.egammaSCAnalyzer_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/d/dlevans/scratch0/CMSSW_2_1_0_pre6/src/reco_fevtsim.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.p = cms.Path(process.egammaBasicClusterAnalyzer+process.egammaSuperClusterAnalyzer)
process.DQM.collectorHost = ''

