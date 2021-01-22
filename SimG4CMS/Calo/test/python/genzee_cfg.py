import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
process.load("SimG4CMS.Calo.PythiaZee_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.Timing = cms.Service("Timing")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('zeer.root')
)

process.p1 = cms.Path(process.generator)
process.outpath = cms.EndPath(process.o1)
