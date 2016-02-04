import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorTest") 
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")


#process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("SimG4CMS.Forward.castorGeometryXML_cfi")
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Configuration.StandardSequences.Generator_cff")


process.RandomNumberGeneratorService.theSource.initialSeed = RAND_sourceSeed
process.RandomNumberGeneratorService.generator.initialSeed = RAND_generator
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = RAND_VtxSmeared
process.RandomNumberGeneratorService.g4SimHits.initialSeed = RAND_g4SimHits
process.RandomNumberGeneratorService.mix.initialSeed = RAND_mix


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(MAXEVENTS)
)


process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MaxEta = cms.double(-5.9),
        MaxPhi = cms.double(3.14),
        MinEta = cms.double(-6.0),
        MinE = cms.double(MINIMUM_ENERGY),
        MinPhi = cms.double(-3.14),
        MaxE = cms.double(MAXIMUM_ENERGY)
    ),
    AddAntiParticle = cms.bool(True),
    Verbosity = cms.untracked.int32(0)
)

process.CaloSD = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold  = cms.double(30.0),
    IonThreshold     = cms.double(30.0),
    SuppressHeavy    = cms.bool(False),
    EminTrack        = cms.double(1.0),
    TmaxHit          = cms.double(1000.0),
    HCNames          = cms.vstring('ZDCHITS'),
    EminHits         = cms.vdouble(0.0),
    TmaxHits         = cms.vdouble(2000.0),
    UseResponseTables= cms.vint32(0),
    BeamPosition     = cms.double(0.0),
    CorrectTOFBeam   = cms.bool(False),
    DetailedTiming   = cms.untracked.bool(False),
    UseMap           = cms.untracked.bool(True),
    Verbosity        = cms.untracked.int32(0),
    CheckHits        = cms.untracked.int32(25)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('POOLOUTPUTFILE')
)


process.common_maximum_timex = cms.PSet( # need to be localy redefined
   MaxTrackTime  = cms.double(500.0),  # need to be localy redefined
   MaxTimeNames  = cms.vstring(), # need to be localy redefined
   MaxTrackTimes = cms.vdouble()  # need to be localy redefined
)
process.p1 = cms.Path(process.generator*process.pgen*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.Generator.ApplyEtaCuts = False



process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   process.common_maximum_timex,        # need to be localy redefined
   TrackNeutrino = cms.bool(False),
   KillDeltaRay  = cms.bool(False),
   KillHeavy     = cms.bool(False),
   SaveFirstLevelSecondary = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
  SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)

process.g4SimHits.SteppingAction = cms.PSet(
   process.common_maximum_timex, # need to be localy redefined
   KillBeamPipe            = cms.bool(True),
   CriticalEnergyForVacuum = cms.double(2.0),
   CriticalDensity         = cms.double(1e-15),
   EkinNames               = cms.vstring(),
   EkinThresholds          = cms.vdouble(),
   EkinParticles           = cms.vstring(),
   Verbosity               = cms.untracked.int32(0)
)

process.g4SimHits.G4Commands = ['/tracking/verbose 1']
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorTestAnalysis'),
    CastorTestAnalysis = cms.PSet(
        EventNtupleFileName = cms.string('NTUPLEFILE'),
        Verbosity = cms.int32(0),
        StepNtupleFileName = cms.string('stepNtuple.root'),
        StepNtupleFlag = cms.int32(0),
        EventNtupleFlag = cms.int32(1)
    )
))

