import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4CMS.CherenkovAnalysis.gun_cff")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("SimG4CMS.CherenkovAnalysis.SingleDREAMXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('files/Cherenkov-e10-a0.root')
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
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CherenkovAnalysis = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
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

process.analyzer = cms.EDFilter("CherenkovAnalysis",
    maxEnergy = cms.double(2.0),
    caloHitSource = cms.InputTag("g4SimHits","EcalHitsEB"),
    nBinsEnergy = cms.uint32(50)
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.analyzer)
process.FlatRandomEGunSource.PGunParameters.MinE = 10.0
process.FlatRandomEGunSource.PGunParameters.MaxE = 10.0
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.ECalSD = cms.PSet(
    TestBeam = cms.untracked.bool(False),
    ReadBothSide = cms.untracked.bool(True),
    BirkL3Parametrization = cms.bool(False),
    doCherenkov = cms.bool(True),
    BirkCut = cms.double(0.1),
    BirkC1 = cms.double(0.013),
    BirkC3 = cms.double(0.0),
    BirkC2 = cms.double(9.6e-06),
    SlopeLightYield = cms.double(0.0),
    UseBirkLaw = cms.bool(False),
    BirkSlope = cms.double(0.253694)
)

