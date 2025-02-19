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
        MinEta = cms.untracked.double(4.0),
        MaxEta = cms.untracked.double(4.0),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinE = cms.untracked.double(300.0),
        MaxE = cms.untracked.double(300.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.load("SimG4Core.GFlash.cmsGflashGeometryXML_cfi")

process.g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
process.g4SimHits.Physics.GFlash = cms.PSet(
    GflashHadronPhysics = cms.string('QGSP_BERT'),
    GflashEMShowerModel = cms.bool(True),
    GflashHadronShowerModel = cms.bool(True),
    GflashHistogram = cms.bool(True),
    GflashHistogramName = cms.string('gflash_hf.root'),
    bField = cms.double(3.8),
    watcherOn = cms.bool(False),
    tuning_pList = cms.vdouble()
)

process.g4SimHits.HCalSD.UseShowerLibrary = False

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_g4_hf.root')
)

process.p1 = cms.Path(process.g4SimHits)
process.outpath = cms.EndPath(process.o1)


