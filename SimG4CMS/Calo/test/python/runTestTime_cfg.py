import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('Physics', 
        'SimG4CoreApplication', 
        'G4cout', 
        'G4cerr'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        Physics = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(5.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-5.5),
        MinE = cms.untracked.double(99.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(100.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring('TrackerDeadRegion', 'CaloRegions'),
    MaxTrackTimes = cms.vdouble(250.0,500.0)
)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.G4Commands = ['/tracking/verbose 1']
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression,
    process.common_maximum_timex,
    TrackNeutrino = cms.bool(False),
    KillHeavy     = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInMuon = cms.untracked.bool(True)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(True),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring('FixedShield01','FixedShield02','FixedShield03','FixedShield04','FixedShield05','FixedShield06','FixedShield07','FixedShield08','FixedShield09','FixedShield10'),
    EkinThresholds          = cms.vdouble(0.1,0.1,10.0,10.0),
    EkinParticles           = cms.vstring('e+','e-','pi+','pi-'),
    Verbosity = cms.untracked.int32(0)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckForHighEtPhotons = cms.untracked.bool(False),
    TrackMin = cms.untracked.int32(0),
    EventStep = cms.untracked.int32(1),
    TrackMax = cms.untracked.int32(0),
    TrackStep = cms.untracked.int32(1),
    VerboseLevel = cms.untracked.int32(0),
    EventMin = cms.untracked.int32(0),
    DEBUG = cms.untracked.bool(False),
    EventMax = cms.untracked.int32(0),
    type = cms.string('TrackingVerboseAction')
))

