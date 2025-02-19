import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mixObjects_cfi import *

process = cms.Process("PRODMIXNEW")
process.load("SimGeneral.MixingModule.mixLowLumPU_mixProdStep1_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/FEAEE71D-D664-DE11-88EA-003048767E51.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.CFWriter = cms.EDProducer("CFWriter",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    
    mixObjects = cms.PSet(
    mixCH = cms.PSet(
      mixCaloHits
    ),
    mixTracks = cms.PSet(
      mixSimTracks
    ),
   mixVertices = cms.PSet(
      mixSimVertices
    ),
    mixSH = cms.PSet(
      mixSimHits
    ),
    mixHepMC = cms.PSet(
      mixHepMCProducts
    )
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIXNEW'),
    fileName = cms.untracked.string('file:/tmp/ebecheva/PCFwriterMixCollTest.root')
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


process.p = cms.Path(process.mix+process.CFWriter)
process.outpath = cms.EndPath(process.out)


