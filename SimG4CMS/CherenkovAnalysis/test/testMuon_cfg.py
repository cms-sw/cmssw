import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("SimG4CMS.CherenkovAnalysis.testMuon_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
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

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(0.0),
        MaxEta = cms.double(0.0),
        MinPhi = cms.double(0.0),
        MaxPhi = cms.double(0.0),
        MinE   = cms.double(200.0),
        MaxE   = cms.double(200.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(0)
)

process.VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    src   = cms.InputTag("generator", "unsmeared"),
    MeanX = cms.double(-12.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaX = cms.double(0.0),
    SigmaY = cms.double(0.0),
    SigmaZ = cms.double(0.0),
    TimeOffset = cms.double(0.0)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalSim=dict()
    process.MessageLogger.CherenkovAnalysis=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.analyzer = cms.EDAnalyzer("XtalDedxAnalysis",
    caloHitSource = cms.InputTag("g4SimHits","HcalHits"),
    EnergyMax = cms.double(200.0)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.analyzer)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.DefaultCutValue = 0.07
process.g4SimHits.StackingAction.SaveFirstLevelSecondary = True
process.g4SimHits.OnlySDs = ['CaloTrkProcessing', 'DreamSensitiveDetector']
process.g4SimHits.ECalSD = cms.PSet(
    TestBeam = cms.untracked.bool(False),
    ReadBothSide = cms.untracked.bool(False),
    BirkL3Parametrization = cms.bool(False),
    DD4Hep = cms.untracked.bool(False),
    doCherenkov = cms.bool(False),
    BirkCut = cms.double(0.1),
    BirkC1 = cms.double(0.0045),
    BirkC3 = cms.double(1.0),
    BirkC2 = cms.double(0.0),
    SlopeLightYield = cms.double(0.0),
    UseBirkLaw = cms.bool(False),
    BirkSlope = cms.double(0.253694)
)

