# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root' )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkGeneric_cff")
process.p =cms.Path(
    process.tauBenchmarkGeneric
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

