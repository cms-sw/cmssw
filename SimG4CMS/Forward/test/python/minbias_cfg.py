import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("SimG4CMS.Calo.PythiaMinBias_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load('Configuration/StandardSequences/Generator_cff')

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.MuonNumbering.muonOffsetESProducer_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('SimG4CoreSensitiveDetector', 
        'SimG4CoreGeometry', 'SimG4CoreApplication', 'MagneticField',
        'ForwardSim', 'TrackerSimInfo',
        'TrackerSimInfoNumbering', 'TrackerMapDDDtoID',
        'CaloSim', 'EcalGeom', 'EcalSim',
        'HCalGeom', 'HcalSim', 'HFShower', 'BscSim'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        SimG4CoreSensitiveDetector = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MagneticField = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ForwardSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TrackerSimInfo = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerSimInfoNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerMapDDDtoID = cms.untracked.PSet(
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
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        BscSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('minbias1_QGSP_BERT_EML.root')
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_minbias1_QGSP_BERT_EML.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.CaloSimHitStudy*process.rndmStore)
process.outpath = cms.EndPath(process.o1)
process.generator.pythiaHepMCVerbosity = False
process.generator.pythiaPylistVerbosity = 0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'
# process.g4SimHits.HCalSD.UseShowerLibrary   = False
# process.g4SimHits.HCalSD.UseParametrize     = True
# process.g4SimHits.HCalSD.UsePMTHits         = True
# process.g4SimHits.HCalSD.UseFibreBundleHits = True
# process.g4SimHits.HFShower.UseShowerLibrary = False
# process.g4SimHits.HFShower.UseHFGflash      = True
# process.g4SimHits.HFShower.TrackEM          = False
# process.g4SimHits.HFShower.OnlyLong         = True
# process.g4SimHits.HFShower.EminLibrary      = 0.0
 
process.g4SimHits.CastorSD.useShowerLibrary = True
process.g4SimHits.CastorSD.minEnergyInGeVforUsingSLibrary = 1.0   # default = 1.0
process.g4SimHits.Generator.MinEtaCut = -7.0

process.common_maximum_timex = cms.PSet( # need to be localy redefined
   MaxTrackTime  = cms.double(500.0),  # need to be localy redefined
   MaxTrackTimeForward = cms.double(2000.0), # ns
   MaxTimeNames  = cms.vstring('ZDCRegion','QuadRegion','InterimRegion'), # need to be localy redefined
   MaxTrackTimes = cms.vdouble(2000.0,0.,0.),  # need to be localy redefined
   MaxZCentralCMS = cms.double(50.0), # m
   DeadRegions   = cms.vstring('QuadRegion','InterimRegion'),
   CriticalEnergyForVacuum = cms.double(2.0),   # MeV
   CriticalDensity         = cms.double(1e-15)  # g/cm3
)

process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   process.common_maximum_timex,        # need to be localy redefined
   KillDeltaRay  = cms.bool(False),
   TrackNeutrino = cms.bool(False),
   KillHeavy     = cms.bool(False),
   KillGamma     = cms.bool(True),
   GammaThreshold = cms.double(0.0001), ## (MeV)
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
   RusRoNeutronEnergyLimit  = cms.double(10.0), ## (MeV)
   RusRoEcalNeutron         = cms.double(0.1),
   RusRoHcalNeutron         = cms.double(0.1),
   RusRoMuonIronNeutron     = cms.double(0.1),
   RusRoPreShowerNeutron    = cms.double(0.1),
   RusRoCastorNeutron       = cms.double(0.1),
   RusRoWorldNeutron        = cms.double(0.1),
   RusRoProtonEnergyLimit  = cms.double(0.0),
   RusRoEcalProton         = cms.double(1.0),
   RusRoHcalProton         = cms.double(1.0),
   RusRoMuonIronProton     = cms.double(1.0),
   RusRoPreShowerProton    = cms.double(1.0),
   RusRoCastorProton       = cms.double(1.0),
   RusRoWorldProton        = cms.double(1.0)
)

process.g4SimHits.SteppingAction = cms.PSet(
   process.common_maximum_timex, # need to be localy redefined
   MaxNumberOfSteps        = cms.int32(50000),
   EkinNames               = cms.vstring(),
   EkinThresholds          = cms.vdouble(),
   EkinParticles           = cms.vstring(),
   Verbosity               = cms.untracked.int32(1)
)

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckForHighEtPhotons = cms.untracked.bool(False),
    TrackMin     = cms.untracked.int32(0),
    TrackMax     = cms.untracked.int32(9999999),
    TrackStep    = cms.untracked.int32(1),
    EventMin     = cms.untracked.int32(0),
    EventMax     = cms.untracked.int32(0),
    EventStep    = cms.untracked.int32(1),
    PDGids       = cms.untracked.vint32(),
    VerboseLevel = cms.untracked.int32(0),
    G4Verbose    = cms.untracked.bool(True),
    DEBUG        = cms.untracked.bool(False),
    type      = cms.string('TrackingVerboseAction')
))
