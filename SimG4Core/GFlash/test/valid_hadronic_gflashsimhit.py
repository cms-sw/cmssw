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
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(1.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-1.0),
        MinE = cms.untracked.double(19.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(20.01)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.load("SimG4Core.GFlash.cmsGflashGeometryXML_cfi")

process.g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
process.g4SimHits.Physics.GFlash = cms.PSet(
process.g4SimHits.Physics.GFlash = cms.PSet(
    GflashHadronPhysics = cms.string('QGSP_BERT'),
    GflashEMShowerModel = cms.bool(True),
    GflashHadronShowerModel = cms.bool(True),
    GflashHistogram = cms.bool(True),
    bField = cms.double(4.0),
    gflash5x5EnergyScale_a = cms.double(1.03551),
    gflash5x5EnergyScale_b = cms.double(0),
    emLateral_pList = cms.vdouble(9.76972e-01, -3.85026e-01, 1.82790e+00, 3.66237e+00)
)

process.g4SimHits.Watchers = cms.VPSet(
    cms.PSet(
      GflashG4Watcher = cms.PSet(
        histFileName = cms.string('gflash_g4Watcher.root')
      ),
    type = cms.string('GflashG4Watcher')
    )
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_gflash_had.root')
)

process.p1 = cms.Path(process.g4SimHits)
process.outpath = cms.EndPath(process.o1)


