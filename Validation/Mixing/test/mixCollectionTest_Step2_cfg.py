import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalVal")
process.load("SimGeneral.MixingModule.mixLowLumPU_mixProdStep2_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load('Configuration.EventContent.EventContent_cff')


process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/FEAEE71D-D664-DE11-88EA-003048767E51.root')
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('test.root')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("Validation.Mixing.mixCollectionValidation_cfi")

process.mix_step = cms.Path(process.mix+process.mixCollectionValidation)
process.end_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.mix_step, process.end_step, process.out_step)

process.mix.input.type = 'fixed'
process.mix.input.nbPileupEvents = cms.PSet(
    averageNumber = cms.double(1.0)
)
process.mix.input.fileNames = cms.untracked.vstring('file:/tmp/ebecheva/PCFwriterMixCollTest.root')

