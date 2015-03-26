import FWCore.ParameterSet.Config as cms

process = cms.Process("egammaAnalysis")
process.load("DQMServices.Core.DQM_cfg")

# End of process
process.load("Configuration.StandardSequences.EndOfProcess_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Validation.EcalClusters.egammaBCAnalyzer_cfi")
process.load("Validation.EcalClusters.egammaSCAnalyzer_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 'file:testfile.root' )
)

process.USER = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_simEcalUnsuppressedDigis_*_*', 
        'drop *_simEcalDigis_*_*', 
        'drop *_simEcalPreshowerDigis_*_*', 
        'drop *_ecalRecHit_*_*', 
        'drop *_ecalPreshowerRecHit_*_*'),
    fileName = cms.untracked.string('TestValidation.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.p = cms.Path(process.egammaBasicClusterAnalyzer+process.egammaSuperClusterAnalyzer+process.endOfProcess+process.USER)
process.DQM.collectorHost = ''

