import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    saveFileName = cms.untracked.string(''),
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(7824367),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    SimEcalTBG4Object = cms.PSet(
        initialSeed = cms.untracked.uint32(12),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simEcalUnsuppressedDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")
process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinE = cms.double(50.001),
        MaxE = cms.double(49.999),
        MinEta = cms.double(-1.0),
        MaxEta = cms.double(1.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    psethack = cms.string('single electron'),
    AddAntiParticle = cms.bool(False),
)

process.load("SimG4Core.GFlash.cmsGflashGeometryXML_cfi")
process.load("SimG4Core.GFlash.GflashSim_cfi")

process.Timing = cms.Service("Timing")

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_gflash_em.root')
)

process.p1 = cms.Path(process.generator*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
