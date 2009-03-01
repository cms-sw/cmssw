# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")



#process.load("Validation.RecoParticleFlow.ztt_cfi")
process.load("Validation.RecoParticleFlow.singletau_cfi")

#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring('file:aod.root'
#                                                              )
#                            )



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkGeneric_cff")

process.p =cms.Path(
    process.tauBenchmarkGeneric
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



process.MessageLogger.cerr.FwkReport.reportEvery = 100

