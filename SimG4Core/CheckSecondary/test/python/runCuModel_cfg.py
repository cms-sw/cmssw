import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("SimG4Core.CheckSecondary.CuTarget_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('SimG4CoreApplication', 'CheckSecondary'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CheckSecondary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        MinEta = cms.double(0.0),
        MaxEta = cms.double(0.0),
        MinPhi = cms.double(1.57079632679),
        MaxPhi = cms.double(1.57079632679),
        PartID = cms.vint32(-211),
#        MinE   = cms.double(1.40694),
#        MaxE   = cms.double(1.40694)
        MinE   = cms.double(5.00195),
        MaxE   = cms.double(5.00195)
#        PartID = cms.vint32(2212),
#        MinE   = cms.double(1.3713),
#        MaxE   = cms.double(1.3713)
#        MinE   = cms.double(1.68535),
#        MaxE   = cms.double(1.68535)
#        MinE   = cms.double(2.2092),
#        MaxE   = cms.double(2.2092)
#        MinE   = cms.double(3.1433),
#        MaxE   = cms.double(3.1433)
#        MinE   = cms.double(5.0873),
#        MaxE   = cms.double(5.0873)
#        MinE   = cms.double(6.0729),
#        MaxE   = cms.double(6.0729)
#        MinE   = cms.double(6.5674),
#        MaxE   = cms.double(6.5674)
#        MinE   = cms.double(7.0626),
#        MaxE   = cms.double(7.0626)
#        MinE   = cms.double(7.5585),
#        MaxE   = cms.double(7.5585)
#        MinE   = cms.double(8.3032),
#        MaxE   = cms.double(8.3032)
#        MinE   = cms.double(9.0488),
#        MaxE   = cms.double(9.0488)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
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
    CutsOnProton    = cms.bool(True),
    Verbosity       = cms.untracked.int32(0),
    EMPhysics       = cms.untracked.bool(False),
    HadPhysics      = cms.untracked.bool(True),
    QuasiElastic    = cms.untracked.bool(True),
    FlagBERT        = cms.untracked.bool(False),
    FlagMuNucl      = cms.bool(False),
    FlagFluo        = cms.bool(False),
    Model           = cms.untracked.string('LEP'),
    type            = cms.string('SimG4Core/Physics/CMSModel'),
    DummyEMPhysics  = cms.bool(False)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckSecondary = cms.PSet(
        SaveInFile = cms.untracked.string('CuBertini65.0GeV.root'),
        Verbosity = cms.untracked.int32(0),
        MinimumDeltaE = cms.untracked.double(0.0),
        KillAfter = cms.untracked.int32(1)
    ),
    type = cms.string('CheckSecondary')
), 
    cms.PSet(
        type = cms.string('KillSecondariesRunAction')
    ))

