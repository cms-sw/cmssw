# Runs PFBenchmarkElecRejectionAnalyzer and PFElecRejectionBenchmark
# on PFTau sample to
# monitor performance of PFTau Electron rejection

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/DC0B8126-DB78-DE11-8962-001D09F25109.root',
        '/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/A0CE6C32-DA78-DE11-96E3-001D09F2532F.root',
        '/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/56BA18F5-E278-DE11-8038-001D09F26509.root',
        '/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/30A1F4B9-DB78-DE11-8F74-001D09F253C0.root'
        #'/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/F0303A91-9278-DE11-AADC-001D09F25456.root',
        #'/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/C6B66CD2-A978-DE11-A5A2-000423D98BC4.root',
        #'/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/A838D597-9078-DE11-AEDD-000423D99896.root',
        #'/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/900FF494-9278-DE11-BA53-001D09F28F1B.root',
        #'/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/4666B35F-9278-DE11-B918-000423D8F63C.root',
        '/store/relval/CMSSW_3_1_2/RelValZEE/GEN-SIM-RECO/MC_31X_V3-v1/0007/0C3128C6-A878-DE11-9CEE-001D09F25208.root'
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkElecRejection_cff")
process.pfTauBenchmarkElecRejection.GenMatchObjectLabel = cms.string('e') #  'tau' or "e"
process.pfTauBenchmarkElecRejection.OutputFile = cms.untracked.string('tauBenchmarkElecRejection_zee.root')

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
