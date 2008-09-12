import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("Configuration.EventContent.EventContent_cff")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("SimG4CMS.CherenkovAnalysis.testMuon_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500000)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_1.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ecHit_1.root')
)

process.Timing = cms.Service("Timing")

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(0.0),
        MaxPhi = cms.untracked.double(0.0),
        MinEta = cms.untracked.double(0.0),
        MinE = cms.untracked.double(200.0),
        MinPhi = cms.untracked.double(0.0),
        MaxE = cms.untracked.double(200.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.VtxSmeared = cms.EDFilter("GaussEvtVtxGenerator",
    MeanX = cms.double(-12.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0),
    SigmaX = cms.double(0.0),
    SigmaZ = cms.double(0.0),
    TimeOffset = cms.double(0.0)
)

process.MessageLogger = cms.Service("MessageLogger",
    errors = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(100)
        )
    ),
    cout = cms.untracked.PSet(
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CherenkovAnalysis = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalSim', 
        'G4cout', 
        'G4cerr', 
        'CherenkovAnalysis', 
        'SimG4CoreApplication', 
        'CaloSim'),
    destinations = cms.untracked.vstring('cout', 
        'errors')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.analyzer = cms.EDFilter("XtalDedxAnalysis",
    caloHitSource = cms.InputTag("g4SimHits","HcalHits"),
    EnergyMax = cms.double(200.0)
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.analyzer)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.Physics.DefaultCutValue = 0.07
process.g4SimHits.StackingAction.SaveFirstLevelSecondary = True
process.g4SimHits.ECalSD = cms.PSet(
    TestBeam = cms.untracked.bool(False),
    ReadBothSide = cms.untracked.bool(False),
    BirkL3Parametrization = cms.bool(False),
    doCherenkov = cms.bool(False),
    BirkCut = cms.double(0.1),
    BirkC1 = cms.double(0.0045),
    BirkC3 = cms.double(1.0),
    BirkC2 = cms.double(0.0),
    SlopeLightYield = cms.double(0.0),
    UseBirkLaw = cms.bool(False),
    BirkSlope = cms.double(0.253694)
)

