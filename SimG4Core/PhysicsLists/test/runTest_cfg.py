import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        PhysicsList = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('PhysicsList'),
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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(3.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.5),
        MinE = cms.untracked.double(5.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(6.01)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics = cms.PSet(
    GFlashEmin = cms.double(1.0),
    G4BremsstrahlungThreshold = cms.double(0.5),
    DefaultCutValue = cms.double(1000.0),
    CutsPerRegion = cms.bool(True),
    Verbosity = cms.untracked.int32(0),
    GFlashEmax = cms.double(1000000.0),
    EMPhysics = cms.untracked.bool(True),
    QuasiElastic = cms.untracked.bool(True),
    GFlashEToKill = cms.double(0.1),
    HadPhysics = cms.untracked.bool(True),
    Model = cms.untracked.string('FTFP'),
    type = cms.string('SimG4Core/Physics/CMSModel'),
    DummyEMPhysics = cms.bool(False)
)

