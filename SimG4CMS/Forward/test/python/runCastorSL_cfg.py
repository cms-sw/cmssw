import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorTest")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration/StandardSequences/Generator_cff')

process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.Geometry.GeometryReco_cff")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ForwardSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),    # 11 => el , 211 => pi
        MinEta = cms.double(-6.2),   # -6.2
        MaxEta = cms.double(-5.6),   # -5.6
        MinPhi = cms.double(0.),     #  0.
        MaxPhi = cms.double(0.7854), #  0.7854
        MinE = cms.double(45.00),
        MaxE = cms.double(45.00)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(0)

)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/sim_pion_SL_1000evts_E45_eta-6.2--5.6_phi0-0.7854_370_NSH_FG_ppONtrkproj.root')
)

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
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 10.
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.CastorSD.useShowerLibrary = True
process.g4SimHits.CastorSD.minEnergyInGeVforUsingSLibrary = 1.0   # default = 1.0
#process.g4SimHits.CastorShowerLibrary.FileName = 'SimG4CMS/Forward/data/castorShowerLibrary.root'
process.g4SimHits.CastorShowerLibrary.BranchEvt = 'hadShowerLibInfo.'
process.g4SimHits.CastorShowerLibrary.BranchEM  = 'emParticles.'
process.g4SimHits.CastorShowerLibrary.BranchHAD = 'hadParticles.'

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
    type = cms.string('CastorTestAnalysis'),
    CastorTestAnalysis = cms.PSet(
        EventNtupleFlag = cms.int32(1),
        StepNtupleFlag  = cms.int32(0),
        EventNtupleFileName = cms.string('eventNtuple.root'),
        StepNtupleFileName  = cms.string('stepNtuple.root'),
        Verbosity = cms.int32(0),
    )
))
