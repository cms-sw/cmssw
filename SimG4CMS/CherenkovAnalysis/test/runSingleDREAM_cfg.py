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
    fileName = cms.string('Cherenkov-e10-a0.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout', 'errors'),
    categories = cms.untracked.vstring(
        'G4cout', 'G4cerr', 'CaloSim', 'CherenkovAnalysis', 
        'EcalSim', 'SimG4CoreApplication'),
    debugModules = cms.untracked.vstring('*'),
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
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CherenkovAnalysis = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.analyzer = cms.EDAnalyzer("CherenkovAnalysis",
    maxEnergy = cms.double(2.0),
    caloHitSource = cms.InputTag("g4SimHits","EcalHitsEB"),
    nBinsEnergy = cms.uint32(50)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.analyzer)
process.generator.PGunParameters.MinE = 10.0
process.generator.PGunParameters.MaxE = 10.0
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
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

