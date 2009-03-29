# Runs PFBenchmarkElecRejectionAnalyzer and PFElecRejectionBenchmark
# on PFTau sample to
# monitor performance of PFTau Electron rejection

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/0C9E3984-57E8-DD11-B89C-001D09F291D2.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/4CBAC9FC-56E8-DD11-90F5-000423D986C4.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/B0AF7439-57E8-DD11-BC2D-001617C3B77C.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/C6AEDDC0-6AE8-DD11-8381-001D09F231B0.root'
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkElecRejection_cff")
process.pfTauBenchmarkElecRejection.GenMatchObjectLabel = cms.string('tau') #  'tau' or "e"
process.pfTauBenchmarkElecRejection.OutputFile = cms.untracked.string('tauBenchmarkElecRejection_ztt.root')

process.p =cms.Path(
    process.tauBenchmarkElecRejection
    )

process.schedule = cms.Schedule(process.p)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)
