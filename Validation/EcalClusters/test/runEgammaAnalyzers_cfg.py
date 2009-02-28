import FWCore.ParameterSet.Config as cms

process = cms.Process("egammaAnalysis")
process.load("DQMServices.Core.DQM_cfg")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Validation.EcalClusters.egammaBCAnalyzer_cfi")
process.load("Validation.EcalClusters.egammaSCAnalyzer_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	#'file:/afs/cern.ch/user/d/dlevans/scratch0/CMSSW_2_1_X_2008-07-01-0000/src/SingleElectronPt35_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
	'/store/relval/2008/6/22/RelVal-RelValSingleElectronPt35-1213987236-IDEAL_V2-2nd/0004/5233133B-C640-DD11-A56C-000423D6CA02.root'

)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.p = cms.Path(process.egammaBasicClusterAnalyzer+process.egammaSuperClusterAnalyzer)
process.DQM.collectorHost = ''

