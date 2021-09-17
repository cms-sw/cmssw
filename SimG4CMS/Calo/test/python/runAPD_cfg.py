import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.EcalTestBeam.APDXML_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.FlatThetaGun=dict()
    process.MessageLogger.EcalGeom=dict()
    process.MessageLogger.EcalSim=dict()

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

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_APD_Epoxy.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runWithAPD_Epoxy.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.CaloSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.VtxSmeared.MeanZ = -1.0
process.VtxSmeared.SigmaX = 0.0
process.VtxSmeared.SigmaY = 0.0
process.VtxSmeared.SigmaZ = 0.0
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyPCuts = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_HP_EML'
process.g4SimHits.Physics.Verbosity = 1
process.g4SimHits.CaloSD.EminHits[0] = 0
process.g4SimHits.ECalSD.NullNumbering  = True
process.g4SimHits.ECalSD.StoreSecondary = True
process.g4SimHits.CaloTrkProcessing.PutHistory = True
process.g4SimHits.G4Commands = ['/run/verbose 2']
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

# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                process.out_step
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
