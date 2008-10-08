import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(11),
        MinEta = cms.untracked.double(-1.0),
        MaxEta = cms.untracked.double(1.0),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinE = cms.untracked.double(20.0),
        MaxE = cms.untracked.double(20.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.load("SimG4Core.GFlash.cmsGflashGeometryXML_cfi")


process.g4SimHits.Watchers = cms.VPSet(
    cms.PSet(
      GflashG4Watcher = cms.PSet(
        histFileName = cms.string('gflash_g4Watcher.root')
      ),
    type = cms.string('GflashG4Watcher')
    )
)


process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_g4_em.root')
)

process.p1 = cms.Path(process.g4SimHits)
process.outpath = cms.EndPath(process.o1)


