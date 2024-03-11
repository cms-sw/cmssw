import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789)
    ),
    sourceSeed = cms.untracked.uint32(54321)
)

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(22),
        MinEta = cms.double(0.2),
        MaxEta = cms.double(0.2),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(30.0),
        MaxE   = cms.double(30.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   
    psethack        = cms.string('30GeV photon on barrel'),
    firstRun        = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Photon_E30GeV_barrel.root')
)

process.p1 = cms.Path(process.generator)
process.p2 = cms.EndPath(process.GEN)

# foo bar baz
