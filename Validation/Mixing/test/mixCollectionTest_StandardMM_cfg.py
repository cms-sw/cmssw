import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalVal")
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load('Configuration.EventContent.EventContent_cff')


process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/FEAEE71D-D664-DE11-88EA-003048767E51.root')
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('mixCollValStandardMM10evs.root')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("Validation.Mixing.mixCollectionValidation_cfi")

process.mix_step = cms.Path(process.mix+process.mixCollectionValidation)
process.end_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.mix_step, process.end_step, process.out_step)

process.mix.input.fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/ECDB1818-A964-DE11-9B4B-001D09F24934.root',
    '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/D245A5BB-4C64-DE11-9F79-001D09F248F8.root',
    '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/C65577F4-EC64-DE11-8D4A-001D09F251CC.root',
    '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/965505C4-9264-DE11-A3BC-001D09F232B9.root',
    '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/5E309A39-7264-DE11-978E-001D09F2A690.root')

