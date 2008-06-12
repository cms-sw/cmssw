import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("SimG4CMS.HcalTestBeam.test.TB2002GeometryXML_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalTBSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        VertexGenerator = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('INFO'),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalGeom', 
        'EcalSim', 
        'HCalGeom', 
        'HcalSim', 
        'HcalTBSim', 
        'FwkJob', 
        'VertexGenerator'),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.common_beam_direction_parameters = cms.PSet(
    MaxEta = cms.untracked.double(0.7397),
    MaxPhi = cms.untracked.double(6.23955),
    MinEta = cms.untracked.double(0.7397),
    MinPhi = cms.untracked.double(6.23955)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        MaxE = cms.untracked.double(20.01),
        MinE = cms.untracked.double(19.99),
        PartID = cms.untracked.vint32(211)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('sim2002.root')
)

process.Tracer = cms.Service("Tracer")

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.VtxSmeared.MeanX = -420.0
process.VtxSmeared.MeanY = 18.338
process.VtxSmeared.MeanZ = -340.11
process.VtxSmeared.SigmaX = 0.000001
process.VtxSmeared.SigmaY = 0.000001
process.VtxSmeared.SigmaZ = 0.000001
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression,
    SuppressHeavy = cms.bool(False),
    DetailedTiming = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0),
    CheckHits = cms.untracked.int32(25),
    CorrectTOFBeam = cms.untracked.bool(False),
    UseMap = cms.untracked.bool(True),
    EminTrack = cms.double(1.0)
)
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('HcalTB02Analysis'),
    HcalTB02Analysis = cms.PSet(
        HistoFileName = cms.untracked.string('HcalTB02Histo.root'),
        Names = cms.vstring('HcalHits', 
            'EcalHitsEB'),
        TupleFileName = cms.untracked.string('HcalTB02Tuple.root'),
        Verbose = cms.untracked.bool(True),
        HcalClusterOnly = cms.untracked.bool(False)
    )
))
process.DQM.collectorHost = ''

