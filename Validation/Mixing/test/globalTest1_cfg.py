import FWCore.ParameterSet.Config as cms

process = cms.Process("GlobalVal")
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.MessageLogger = cms.Service("MessageLogger")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/RelVal/2007/7/11/RelVal-RelVal160pre4SingleEPt35-1184176348/0000/5EF3794C-7530-DC11-833F-000423D6C8EE.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *CrossingFrame_*_*_*'),
    fileName = cms.untracked.string('/tmp/Cum_global.root')
)

process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.out)
process.mix.input.type = 'fixed'
process.mix.input.nbPileupEvents = cms.PSet(
    averageNumber = cms.double(20.0)
)


