# This is an example how to run with 4 sources, and in playback mode

import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODMIXBack")
process.load("SimGeneral.MixingModule.mixLowLumPU_4sources_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(77)
    )
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(2),
    fileNames = cms.untracked.vstring('file:Cum_store_4s.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIX'),
    fileName = cms.untracked.string('file:Cum_restored_4s.root')
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
process.mix.playback = True


