# Runs PFBenchmarkAnalyzer and PFElectronBenchmark on PFElectron sample to
# monitor performance of PFElectron

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
		            fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre8/RelValZEE/GEN-SIM-RECO/STARTUP_31X_v1/0006/443872D1-DA4D-DE11-87DB-000423D94700.root',
                                                              '/store/relval/CMSSW_3_1_0_pre8/RelValZEE/GEN-SIM-RECO/STARTUP_31X_v1/0005/C258C2F4-5A4D-DE11-B2EF-001D09F2424A.root',
                                                              '/store/relval/CMSSW_3_1_0_pre8/RelValZEE/GEN-SIM-RECO/STARTUP_31X_v1/0005/3463063D-614D-DE11-8717-001D09F242EA.root',
                                                              '/store/relval/CMSSW_3_1_0_pre8/RelValZEE/GEN-SIM-RECO/STARTUP_31X_v1/0005/0AC4EE74-5F4D-DE11-A410-001D09F23944.root')

			    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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

