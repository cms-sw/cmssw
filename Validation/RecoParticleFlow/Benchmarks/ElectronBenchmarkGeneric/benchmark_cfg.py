# Runs PFBenchmarkAnalyzer and PFElectronBenchmark on PFElectron sample to
# monitor performance of PFElectron

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
#		            fileNames = cms.untracked.vstring('file:/localscratch/b/beaudett/validation/CMSSW_3_1_0/src/RecoEgamma/Examples/test/SingleElectrons_Fast.root'),
                            fileNames = cms.untracked.vstring('file:/localscratch/b/beaudett/pflow/CMSSW_3_1_1/src/FastSimulation/Configuration/test/AODIntegrationTestWithHLT.root'),			    
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
                            )

process.load("FWCore.Modules.printContent_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("PhysicsTools.PFCandProducer.pfElectrons_cff")

process.load("Validation.RecoParticleFlow.electronBenchmarkGeneric_cff")

process.gensource.select = cms.vstring(
    "drop *",
    "keep+ pdgId = 24",
    "keep+ pdgId = -24",
    "drop pdgId !=11 && pdgId !=-11"
    )

process.p =cms.Path(
#    process.printContent
    process.electronBenchmarkGeneric    
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

