# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
                            #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root' )
                            #fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre4/RelValZEE/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/16D601C5-7416-DE11-A0B8-001A92971AAA.root' )
                            #fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre4/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/12EEDF82-6516-DE11-A513-001A928116D2.root' )
                            #fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre6/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0002/FEAA5C48-1733-DE11-9F90-000423D9997E.root',
        		    #           		       '/store/relval/CMSSW_3_1_0_pre6/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0002/CC9CB83E-A132-DE11-B2DA-001617C3B78C.root',
                            #                                  '/store/relval/CMSSW_3_1_0_pre6/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0002/90863F54-A232-DE11-B8DF-001617E30D0A.root',
                            #                                  '/store/relval/CMSSW_3_1_0_pre6/RelValZEE/GEN-SIM-RECO/IDEAL_31X_v1/0002/4E2A7DD9-A032-DE11-942E-001617C3B66C.root')
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

