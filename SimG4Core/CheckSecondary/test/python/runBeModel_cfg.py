import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("SimG4Core.CheckSecondary.BeTarget_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CheckSecondary = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('SimG4CoreGeometry', 
        'CheckSecondary'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(2212),
        MaxEta = cms.untracked.double(0.0),
        MaxPhi = cms.untracked.double(1.57079632679),
        MinEta = cms.untracked.double(0.0),
        MinE = cms.untracked.double(8.2535),
        MinPhi = cms.untracked.double(1.57079632679),
        MaxE = cms.untracked.double(8.2535)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics = cms.PSet(
    GFlash          = cms.PSet(
      GflashHistogram = cms.bool(False),
      GflashEMShowerModel = cms.bool(False),
      GflashHadronPhysics = cms.string('QGSP_BERT_EMV'),
      GflashHadronShowerModel = cms.bool(False)
    ),
    G4BremsstrahlungThreshold = cms.double(0.5),
    DefaultCutValue = cms.double(1.0),
    CutsPerRegion   = cms.bool(True),
    Verbosity       = cms.untracked.int32(0),
    EMPhysics       = cms.untracked.bool(False),
    HadPhysics      = cms.untracked.bool(True),
    QuasiElastic    = cms.untracked.bool(True),
    FlagBERT        = cms.untracked.bool(False),
    FlagCHIPS       = cms.untracked.bool(False),
    FlagFTF         = cms.untracked.bool(False),
    FlagGlauber     = cms.untracked.bool(False),
    Model           = cms.untracked.string('Bertini'),
    type            = cms.string('SimG4Core/Physics/CMSModel'),
    DummyEMPhysics  = cms.bool(False)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckSecondary = cms.PSet(
        SaveInFile = cms.untracked.string('BeBertini8.2GeV.root'),
        Verbosity = cms.untracked.int32(0),
        MinimumDeltaE = cms.untracked.double(0.0),
        KillAfter = cms.untracked.int32(1)
    ),
    type = cms.string('CheckSecondary')
), 
    cms.PSet(
        type = cms.string('KillSecondariesRunAction')
    ))

