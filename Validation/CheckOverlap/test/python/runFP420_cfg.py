import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('G4cout', 'G4cerr'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
)

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
        MaxEta = cms.untracked.double(3.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.5),
        MinE = cms.untracked.double(9.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type       = cms.string('CheckOverlap'),
    Resolution = cms.untracked.int32(1000),
    NodeNames  = cms.untracked.vstring('FP420E')
))

