import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("SimG4CMS.HcalTestBeam.TB2006GeometryXML_cfi")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hcaltb06.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalGeom', 
        'EcalSim', 
        'HCalGeom', 
        'HcalSim', 
        'HcalTBSim', 
        'SimHCalData', 
        'SimG4CoreGeometry', 
        'SimG4CoreApplication', 
        'VertexGenerator'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        VertexGenerator = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalTBSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimHCalData = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.common_beam_direction_parameters = cms.PSet(
    MinEta       = cms.double(0.5655),
    MaxEta       = cms.double(0.5655),
    MinPhi       = cms.double(-0.1309),
    MaxPhi       = cms.double(-0.1309),
    BeamPosition = cms.double(-800.0)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        MinE   = cms.double(50.0),
        MaxE   = cms.double(50.0),
        PartID = cms.vint32(-211)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('sim2006.root')
)

process.common_heavy_suppression1 = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)

process.Timing = cms.Service("Timing")

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamMeanX       = cms.double(0.0),
    BeamMeanY       = cms.double(0.0),
    BeamSigmaX      = cms.double(0.0001),
    BeamSigmaY      = cms.double(0.0001),
    Psi             = cms.double(999.9),
    GaussianProfile = cms.bool(False),
    BinX            = cms.int32(50),
    BinY            = cms.int32(50),
    File            = cms.string('beam.profile'),
    UseFile         = cms.bool(False),
    TimeOffset      = cms.double(0.)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble()
)
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.Region = 'HcalRegion'
process.g4SimHits.Physics.DefaultCutValue = 1.

process.g4SimHits.ECalSD.UseBirkLaw = True
process.g4SimHits.ECalSD.BirkL3Parametrization = True
process.g4SimHits.ECalSD.BirkC1 = 0.033
process.g4SimHits.ECalSD.BirkC2 = 0.0
process.g4SimHits.ECalSD.SlopeLightYield = 0.02
process.g4SimHits.HCalSD.UseBirkLaw = True
process.g4SimHits.HCalSD.BirkC1 = 0.0052
process.g4SimHits.HCalSD.BirkC2 = 0.142
process.g4SimHits.HCalSD.BirkC3 = 1.75
process.g4SimHits.HCalSD.UseLayerWt = False
process.g4SimHits.HCalSD.WtFile     = ' '
process.g4SimHits.HCalSD.UseShowerLibrary    = False
process.g4SimHits.HCalSD.TestNumberingScheme = False
process.g4SimHits.HCalSD.UseHF   = False
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression1,
    process.common_maximum_timex,
    TrackNeutrino = cms.bool(False),
    KillDeltaRay  = cms.bool(False),
    KillHeavy     = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInCalo = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInMuon = cms.untracked.bool(False),
    RusRoEcalNeutron         = cms.double(1.0),
    RusRoEcalNeutronLimit    = cms.double(0.0),
    RusRoHcalNeutron         = cms.double(1.0),
    RusRoHcalNeutronLimit    = cms.double(0.0),
    RusRoEcalProton      = cms.double(1.0),
    RusRoEcalProtonLimit = cms.double(0.0),
    RusRoHcalProton      = cms.double(1.0),
    RusRoHcalProtonLimit = cms.double(0.0)
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
process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression1,
    EminTrack      = cms.double(1.0),
    TmaxHit        = cms.double(1000.0),
    EminHits       = cms.vdouble(0.0,0.0,0.0,0.0),
    EminHitsDepth  = cms.vdouble(0.0,0.0,0.0,0.0),
    TmaxHits       = cms.vdouble(1000.0,1000.0,1000.0,1000.0),
    HCNames        = cms.vstring('EcalHitsEB','EcalHitsEE','EcalHitsES','HcalHits'),
    UseResponseTables = cms.vint32(0,0,0,0),
    SuppressHeavy  = cms.bool(False),
    CheckHits      = cms.untracked.int32(25),
    UseMap         = cms.untracked.bool(True),
    Verbosity      = cms.untracked.int32(0),
    DetailedTiming = cms.untracked.bool(False),
    CorrectTOFBeam = cms.bool(False)
)
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HcalTB06Analysis = cms.PSet(
        process.common_beam_direction_parameters,
        Names    = cms.vstring('HcalHits', 'EcalHitsEB'),
        EHCalMax = cms.untracked.double(10.0),
        ETtotMax = cms.untracked.double(500.0),
        Verbose  = cms.untracked.bool(True)
    ),
    type = cms.string('HcalTB06Analysis')
))
