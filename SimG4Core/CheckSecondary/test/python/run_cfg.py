import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CheckSecondary = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('SimG4CoreApplication', 
        'CheckSecondary'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(1.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-1.5),
        MinE = cms.untracked.double(9.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.op = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simevent.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.op)
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble()
)
process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_EMV'
process.g4SimHits.StackingAction = cms.PSet(
    process.common_maximum_timex,
    TrackNeutrino    = cms.bool(False),
    KillHeavy        = cms.bool(True),
    NeutronThreshold = cms.double(100.0),
    ProtonThreshold  = cms.double(100.0),
    IonThreshold     = cms.double(100.0),
    SaveFirstLevelSecondary = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInMuon = cms.untracked.bool(False)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(True),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity = cms.untracked.int32(0)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    StoreSecondary = cms.PSet(
        Verbosity = cms.untracked.int32(2),
        MinimumDeltaE = cms.untracked.double(50.0),
        KillAfter = cms.untracked.int32(1)
    ),
    type = cms.string('StoreSecondary')
))

