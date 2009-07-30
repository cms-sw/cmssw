import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
process.load("SimG4CMS.Calo.PythiaMinBias_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            )
    )
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('minbias.root')
)

process.p1 = cms.Path(process.generator)
process.outpath = cms.EndPath(process.o1)
