import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mixObjects_cfi import *


process = cms.Process("PRODMIXNEW")
process.load("SimGeneral.MixingModule.mixLowLumPU_mixProdStep2_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/data/becheva/MixingModule/dataFiles/relval/02C5A172-8203-DE11-86D7-001617C3B5D8TTBar.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIXNEW'),
    fileName = cms.untracked.string('file:/tmp/ebecheva/testMixProdStep2.root')
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
