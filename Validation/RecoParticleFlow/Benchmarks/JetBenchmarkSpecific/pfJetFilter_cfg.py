# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms
  
process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source (
    "PoolSource",    
    fileNames = cms.untracked.vstring(
    # Fast
       #'file:../../test/aod_QCDForPF_Fast_0.root'
     # Full
       'file:../../test/aod_QCDForPF_Full_001.root',
       'file:../../test/aod_QCDForPF_Full_002.root',
       'file:../../test/aod_QCDForPF_Full_003.root'
       ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Validation.RecoParticleFlow.PFJetFilter_cff")

process.p =cms.Path(process.pfFilter)

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

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
