# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
                            #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root' )
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/16D601C5-7416-DE11-A0B8-001A92971AAA.root' )
                            #fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre4/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/12EEDF82-6516-DE11-A513-001A928116D2.root' )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(300)
)


process.load("Validation.RecoParticleFlow.electronBenchmarkGeneric_cff")
process.p =cms.Path(
    process.electronBenchmarkGeneric
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

