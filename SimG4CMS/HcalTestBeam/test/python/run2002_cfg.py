import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_h2tb_cff import h2tb

process = cms.Process("PROD", h2tb)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("SimG4CMS.HcalTestBeam.TB2002GeometryXML_cfi")
process.load("Geometry.HcalTestBeamData.hcalDDDSimConstants_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hcaltb02.root')
)

process.load('FWCore.MessageService.MessageLogger_cfi')
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalTBSim=dict()
    process.MessageLogger.VertexGenerator=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.common_beam_direction_parameters = cms.PSet(
    MinEta       = cms.double(0.7397),
    MaxEta       = cms.double(0.7397),
    MinPhi       = cms.double(6.23955),
    MaxPhi       = cms.double(6.23955),
    BeamPosition = cms.double(0.0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("EmptySource",
    firstRun   = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        MinE   = cms.double(19.99),
        MaxE   = cms.double(20.01),
        PartID = cms.vint32(211)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('sim2002.root')
)

process.Tracer = cms.Service("Tracer")

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.VtxSmeared.MeanX = -420.0
process.VtxSmeared.MeanY = 18.338
process.VtxSmeared.MeanZ = -340.11
process.VtxSmeared.SigmaX = 0.000001
process.VtxSmeared.SigmaY = 0.000001
process.VtxSmeared.SigmaZ = 0.000001
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.OnlySDs = ['CaloTrkProcessing',
                             'EcalTBH4BeamDetector',
                             'HcalTB02SensitiveDetector',
                             'HcalTB06BeamDetector',
                             'EcalSensitiveDetector',
                             'HcalSensitiveDetector']
process.g4SimHits.CaloSD.EminHits = [0.0,0.0,0.0,0.0]
process.g4SimHits.CaloSD.TmaxHits = [1000.0,1000.0,1000.0,1000.0]
process.g4SimHits.CaloSD.UseMap = True
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('HcalTB02Analysis'),
    HcalTB02Analysis = cms.PSet(
        Names           = cms.vstring('HcalHits', 'EcalHitsEB'),
        HcalClusterOnly = cms.untracked.bool(False),
        Verbose         = cms.untracked.bool(True)
    )
))

