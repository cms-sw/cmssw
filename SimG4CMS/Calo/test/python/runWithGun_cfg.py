import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimG4CMS.Calo.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.EcalSim=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-3.0),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt  = cms.double(100.),
        MaxPt  = cms.double(100.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_QGSP_FTFP_BERT_EML.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

#process.Tracer = cms.Service("Tracer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runWithGun_FTFP_BERT_EMM.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.CaloSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.CaloSimHitStudy.MaxEnergy = 1000.0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Physics.MonopoleCharge = 1
process.g4SimHits.Physics.Verbosity = 0
process.g4SimHits.CaloSD.UseResponseTables = [1,1,0,1]
process.g4SimHits.CaloSD.EminHits[0] = 0
process.g4SimHits.ECalSD.StoreSecondary = True
process.g4SimHits.ECalSD.StoreRadLength = True
process.g4SimHits.ECalSD.ScaleRadLength = 100.0
process.g4SimHits.CaloTrkProcessing.PutHistory = True
process.g4SimHits.CaloResponse.UseResponseTable  = True
process.g4SimHits.CaloResponse.ResponseScale = 1.0
process.g4SimHits.CaloResponse.ResponseFile = 'SimG4CMS/Calo/data/responsTBpim50.dat'
process.g4SimHits.G4Commands = ['/run/verbose 2']
process.CaloSimHitStudy.StoreRL = True
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
    TrackNeutrino = cms.bool(True),
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

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                process.out_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
