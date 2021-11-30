import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = dict(
        generator = 456789
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.maxEvents = dict(
        input = 10000
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = dict(
        PartID = 211,
        MinEta = -3.0,
        MaxEta = 3.0,
        MinPhi = -3.14159265359,
        MaxPhi = 3.14159265359,
        MinPt  = 60.0,
        MaxPt  = 60.0
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack        = cms.string('scan with pion of pt=60GeV'),
    firstRun        = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Pion_Pt60GeV_all.root')
)

process.p1 = cms.Path(process.generator)
process.p2 = cms.EndPath(process.GEN)

