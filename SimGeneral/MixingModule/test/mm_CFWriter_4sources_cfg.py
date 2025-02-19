import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.mixObjects_cfi import *

process = cms.Process("PRODMIXNEW")
process.load("SimGeneral.MixingModule.mixLowLumPU_4sources_mixProdStep1_cfi")

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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/FA4BC00A-946B-DE11-9AE4-000423D9939C.root',
        '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/706FCB9B-906B-DE11-959D-000423D991F0.root',
        '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/64060CE2-926B-DE11-90E5-000423D99A8E.root',
        '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/426F0F2B-8C6B-DE11-9CA6-000423D991F0.root',
        '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/3C9F6B26-D86B-DE11-AAB2-001D09F2432B.root'	
)
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
    fileName = cms.untracked.string('file:/tmp/ebecheva/MixedData311_4sources.root')
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


