# Runs PFBenchmarkElecRejectionAnalyzer and PFElecRejectionBenchmark
# on PFTau sample to
# monitor performance of PFTau Electron rejection

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_0_0_pre7/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0006/1A53606D-57E8-DD11-A649-000423D9A2AE.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0006/E0A459F8-6AE8-DD11-8D14-000423D99896.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0006/F2F29E16-57E8-DD11-87DF-001D09F23E53.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZEE/GEN-SIM-RECO/STARTUP_30X_v1/0006/F8F77B9C-57E8-DD11-A065-001D09F24D8A.root'
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkElecRejection_cff")
process.pfTauBenchmarkElecRejection.GenMatchObjectLabel = cms.string('e') #  'tau' or "e"
process.pfTauBenchmarkElecRejection.OutputFile = cms.untracked.string('tauBenchmarkElecRejection.root')

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
