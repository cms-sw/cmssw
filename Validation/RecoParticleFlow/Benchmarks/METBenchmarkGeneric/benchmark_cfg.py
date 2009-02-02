# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_000.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_001.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_002.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_003.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_004.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_005.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_006.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_007.root',
'file:/uscms_data/d2/cavanaug/DATA/CMSSW300pre6/aod_QCDForPF_Full_008.root'
)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Validation.RecoParticleFlow.metBenchmarkGeneric_cff")

process.p =cms.Path(
    process.metBenchmarkGeneric
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

process.MessageLogger.cerr.FwkReport.reportEvery = 50
