import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MagneticField = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
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
        'HFShower', 
        'SimG4CoreApplication', 
        'G4cout', 
        'G4cerr', 
        'MagneticField'),
    destinations = cms.untracked.vstring('cout')
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(-5.5),
        MaxEta = cms.double(5.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(99.99),
        MaxE   = cms.double(100.01)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('PionHcalTestAnalysis.root')
)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'
process.g4SimHits.G4Commands = ['/tracking/verbose 1']
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring('TrackerDeadRegion', 'CaloRegions'),
    MaxTrackTimes = cms.vdouble(250.0,500.0)
)
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression,
    process.common_maximum_timex,
    KillDeltaRay  = cms.bool(False),
    TrackNeutrino = cms.bool(False),
    KillHeavy     = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(True),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity               = cms.untracked.int32(0)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HcalQie = cms.PSet(
        NumOfBuckets = cms.int32(10),
        BaseLine = cms.int32(4),
        BinOfMax = cms.int32(6),
        PreSamples = cms.int32(0),
        EDepPerPE = cms.double(0.0005),
        SignalBuckets = cms.int32(2),
        SigmaNoise = cms.double(0.5),
        qToPE = cms.double(4.0)
    ),
    type = cms.string('HcalTestAnalysis'),
    HcalTestAnalysis = cms.PSet(
        Eta0 = cms.double(0.0),
        LayerGrouping = cms.int32(1),
        FileName = cms.string('PionHcalTestAnalysis.root'),
        Names = cms.vstring('HcalHits',
            'EcalHitsEB',
            'EcalHitsEE'),
        CentralTower = cms.int32(404),
        Phi0 = cms.double(0.0)
    )
))
