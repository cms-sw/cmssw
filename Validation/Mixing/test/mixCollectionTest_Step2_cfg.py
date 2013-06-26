import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalVal")
process.load("SimGeneral.MixingModule.mixLowLumPU_mixProdStep2_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load('Configuration.EventContent.EventContent_cff')


process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EA8E5AF7-576B-DE11-BA98-001D09F24498.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/E8627E8B-5A6B-DE11-A8F4-001D09F2438A.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/D66DD273-5C6B-DE11-A8DB-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/CC3232F2-596B-DE11-8C47-0019B9F704D6.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/AAFDF230-5C6B-DE11-BF0A-001D09F24498.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/9A28D939-576B-DE11-811D-000423D944F0.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94F72FC0-5B6B-DE11-8215-000423D6AF24.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94710927-5B6B-DE11-92EA-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/88D87820-5B6B-DE11-B522-0019B9F704D6.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7E7FB3BC-E16B-DE11-9374-000423D8F63C.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7640E138-576B-DE11-B907-000423D99AAE.root'	
)
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('mixCollValStep2.root')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('mix'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MixingModule = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        )
    ),
    categories = cms.untracked.vstring('MixingModule'),
    destinations = cms.untracked.vstring('cout')
)
process.load("Validation.Mixing.mixCollectionValidation_Step2_cfi")

process.mix_step = cms.Path(process.mix+process.mixCollectionValidation)
process.end_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.mix_step, process.end_step, process.out_step)

process.mix.input.fileNames = cms.untracked.vstring('file:MixedSources.root')
