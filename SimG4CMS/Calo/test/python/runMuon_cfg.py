import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimG4CMS.Calo.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalGeom', 
        'EcalSim', 
        'HCalGeom', 
        'HcalSim', 
        'HFShower', 
        'SimG4CoreApplication', 
        'G4cout', 
        'G4cerr', 
        'HitStudy'),
#    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
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
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(0.1309),
        MaxEta = cms.double(0.1309),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE  = cms.double(50.),
        MaxE  = cms.double(50.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simeventMuon.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runMuon.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.caloSimHitStudy)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'
process.g4SimHits.Physics.Verbosity = 0
process.g4SimHits.CaloSD.UseResponseTables = [1,1,0,1]
process.g4SimHits.CaloResponse.UseResponseTable  = True
process.g4SimHits.CaloResponse.ResponseScale = 1.0
process.g4SimHits.CaloResponse.ResponseFile = 'SimG4CMS/Calo/data/responsTBpim50.dat'
process.g4SimHits.G4Commands = ['/tracking/verbose 1']
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble(),
    DeadRegions   = cms.vstring(),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15)
)
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression,
    process.common_maximum_timex,
    TrackNeutrino = cms.bool(False),
    KillDeltaRay  = cms.bool(False),
    KillHeavy     = cms.bool(False),
    KillGamma     = cms.bool(False),
    GammaThreshold= cms.double(0.0001),  ## (MeV)
    SaveFirstLevelSecondary = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True),
        RusRoGammaEnergyLimit  = cms.double(5.0), ## (MeV)
        RusRoEcalGamma         = cms.double(0.3),
        RusRoHcalGamma         = cms.double(0.3),
        RusRoMuonIronGamma     = cms.double(0.3),
        RusRoPreShowerGamma    = cms.double(0.3),
        RusRoCastorGamma       = cms.double(0.3),
        RusRoWorldGamma        = cms.double(0.3),
        RusRoNeutronEnergyLimit= cms.double(10.0), ## (MeV)
        RusRoEcalNeutron       = cms.double(0.1),
        RusRoHcalNeutron       = cms.double(0.1),
        RusRoMuonIronNeutron   = cms.double(0.1),
        RusRoPreShowerNeutron  = cms.double(0.1),
        RusRoCastorNeutron     = cms.double(0.1),
        RusRoWorldNeutron      = cms.double(0.1),
        RusRoProtonEnergyLimit = cms.double(0.0),
        RusRoEcalProton        = cms.double(1.0),
        RusRoHcalProton        = cms.double(1.0),
        RusRoMuonIronProton    = cms.double(1.0),
        RusRoPreShowerProton   = cms.double(1.0),
        RusRoCastorProton      = cms.double(1.0),
        RusRoWorldProton       = cms.double(1.0)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity               = cms.untracked.int32(2)
)
#process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
#    CheckForHighEtPhotons = cms.untracked.bool(False),
#    TrackMin     = cms.untracked.int32(0),
#    TrackMax     = cms.untracked.int32(0),
#    TrackStep    = cms.untracked.int32(1),
#    EventMin     = cms.untracked.int32(0),
#    EventMax     = cms.untracked.int32(0),
#    EventStep    = cms.untracked.int32(1),
#    PDGids       = cms.untracked.vint32(),
#    VerboseLevel = cms.untracked.int32(0),
#    G4Verbose    = cms.untracked.bool(True),
#    DEBUG        = cms.untracked.bool(False),
#    type      = cms.string('TrackingVerboseAction')
#))

