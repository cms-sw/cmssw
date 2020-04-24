import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODMIXNEW")
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#    oncePerEventMode = cms.untracked.bool(True),
#    ignoreTotal = cms.untracked.int32(1)
#)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0004/F69B70C0-FB73-DF11-A055-002354EF3BD0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIXNEW'),
    fileName = cms.untracked.string('file:/tmp/Cum_store.root')
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

process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.out)

process.e = cms.EndPath(process.out)

process.mix.input.type = 'histo'
process.mix.input.nbPileupEvents = cms.PSet(
    fileName = cms.untracked.string('histrandom.root'),
    histoName = cms.untracked.string('h1f'),
)

process.mix.input.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0004/746D2F6B-1F74-DF11-B664-001A928116B0.root',
       '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/F2181911-EE73-DF11-86BC-0030486791BA.root',
       '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/F063C904-F273-DF11-831C-001A92971B64.root',
       '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/D4B25A5D-F373-DF11-99E7-00304867918E.root',
       '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/8AA2EF2A-F073-DF11-82F7-001A92971B32.root'
)
