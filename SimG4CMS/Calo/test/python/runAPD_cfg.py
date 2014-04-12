import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.EcalTestBeam.APDXML_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('CaloSim', 'EcalGeom', 'EcalSim', 
                                       'SimG4CoreApplication', 'FlatThetaGun',
                                       'G4cout', 'G4cerr', 'SimTrackManager'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
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
        SimTrackManager = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FlatThetaGun = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FileRandomKEThetaGunProducer",
    PGunParameters = cms.PSet(
        PartID   = cms.vint32(2112),
        MinTheta = cms.double(0.0),
        MaxTheta = cms.double(0.0),
        MinPhi   = cms.double(-3.14159265359),
        MaxPhi   = cms.double(3.14159265359),
        Particles= cms.int32(1000),
        File     = cms.FileInPath('SimG4CMS/Calo/data/neutronFromCf.dat')
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_APD_Epoxy.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runWithAPD_Epoxy.root')
)

process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble()
)
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.caloSimHitStudy)
process.VtxSmeared.MeanZ = -1.0
process.VtxSmeared.SigmaX = 0.0
process.VtxSmeared.SigmaY = 0.0
process.VtxSmeared.SigmaZ = 0.0
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyPCuts = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_HP'
process.g4SimHits.Physics.Verbosity = 1
process.g4SimHits.CaloSD.EminHits[0] = 0
process.g4SimHits.ECalSD.NullNumbering  = True
process.g4SimHits.ECalSD.StoreSecondary = True
process.g4SimHits.CaloTrkProcessing.PutHistory = True
process.g4SimHits.G4Commands = ['/run/verbose 2']
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression,
    process.common_maximum_timex,
    KillDeltaRay  = cms.bool(True),
    TrackNeutrino = cms.bool(False),
    KillHeavy     = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(False),
    CriticalEnergyForVacuum = cms.double(0.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity               = cms.untracked.int32(2)
)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckForHighEtPhotons = cms.untracked.bool(False),
    EventMin  = cms.untracked.int32(0),
    EventMax  = cms.untracked.int32(3),
    EventStep = cms.untracked.int32(1),
    TrackMin  = cms.untracked.int32(0),
    TrackMax  = cms.untracked.int32(999999999),
    TrackStep = cms.untracked.int32(1),
    VerboseLevel = cms.untracked.int32(2),
    PDGids    = cms.untracked.vint32(),
    G4Verbose = cms.untracked.bool(True),
    DEBUG     = cms.untracked.bool(False),
    type      = cms.string('TrackingVerboseAction')
))

