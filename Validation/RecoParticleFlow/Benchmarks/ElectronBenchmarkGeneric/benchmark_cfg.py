# Runs PFBenchmarkAnalyzer and PFElectronBenchmark on PFElectron sample to
# monitor performance of PFElectron

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
		            fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre7/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0004/A083C68A-E641-DE11-9169-001D09F28F0C.root',
                                                              '/store/relval/CMSSW_3_1_0_pre7/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0004/9C5A4F6B-AF41-DE11-9368-000423D99B3E.root',
                                                              '/store/relval/CMSSW_3_1_0_pre7/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0004/5AA9DD79-B041-DE11-8CDA-001D09F2503C.root',
                                                              '/store/relval/CMSSW_3_1_0_pre7/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0004/347A0DAC-B241-DE11-B892-001D09F2B2CF.root')

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

