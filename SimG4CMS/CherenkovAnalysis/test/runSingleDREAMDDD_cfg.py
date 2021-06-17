import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4CMS.CherenkovAnalysis.gun_cff")
process.load("SimG4CMS.CherenkovAnalysis.SingleDREAMXML_cfi")
process.load("Geometry.HcalCommonData.caloSimulationParameters_cff")
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load("SimG4Core.Application.g4SimHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('Cherenkov-e10-a0-ddd.root')
)


process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalSim=dict()
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.CherenkovAnalysis=dict()
    process.MessageLogger.SimG4CoreGeometry=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.analyzer = cms.EDAnalyzer("CherenkovAnalysis",
    maxEnergy = cms.double(2.0),
    caloHitSource = cms.InputTag("g4SimHits","EcalHitsEB"),
    nBinsEnergy = cms.uint32(50)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.analyzer)

process.generator.PGunParameters.MinE = 10.0
process.generator.PGunParameters.MaxE = 10.0
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.OnlySDs = ['CaloTrkProcessing', 'DreamSensitiveDetector']
process.g4SimHits.ECalSD = cms.PSet(
    TestBeam = cms.untracked.bool(False),
    ReadBothSide = cms.untracked.bool(True),
    DD4Hep = cms.untracked.bool(False),
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

