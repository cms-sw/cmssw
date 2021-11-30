import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = dict(
        generator = 456789
    ),
    sourceSeed = 54321
)

process.maxEvents = dict(
        input = 10000
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = dict(
        PartID = 22,
        MinEta = -3.0,
        MaxEta = 3.0,
        MinPhi = -3.14159265359,
        MaxPhi = 3.14159265359,
        MinE   = 30.0,
        MaxE   = 30.0
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    psethack        = cms.string('scan with 30GeV photon'),
    firstRun        = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Photon_E30GeV_all.root')
)

process.p1 = cms.Path(process.generator)
process.p2 = cms.EndPath(process.GEN)

