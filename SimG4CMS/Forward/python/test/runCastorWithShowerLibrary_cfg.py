import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorTest")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("SimG4CMS.Forward.castorGeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.RandomNumberGeneratorService.generator.initialSeed = 113456789

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ForwardSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(-6.6),
        MaxEta = cms.double(-5.2),
        MinPhi = cms.double(-3.14),
        MaxPhi = cms.double(3.14),
        MinE = cms.double(50.00),
        MaxE = cms.double(50.00)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(1)

)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/sim_pion_SL.root')
)

process.common_maximum_timex = cms.PSet( # need to be localy redefined
   MaxTrackTime  = cms.double(500.0),  # need to be localy redefined
   MaxTimeNames  = cms.vstring(), # need to be localy redefined
   MaxTrackTimes = cms.vdouble()  # need to be localy redefined
)
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 10.
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.CastorSD.useShowerLibrary = True
process.g4SimHits.CastorSD.minEnergyInGeVforUsingSLibrary = 1.0   # default = 1.0
process.g4SimHits.CastorShowerLibrary.FileName = '../../../../../../p/polme/scratch0/CMSSW_3_7_0/src/SL_em+had_E1-1.5-2-2.5-3-3.5-4-4.5-5-6-7-8-9-10-12.5-15-17.5-20-25-30-35-40-45-50-60-70-80-100-125-150-175-200-300-400-500GeV_7eta-6.6--5.2_5phi0-0.7854.root'
process.g4SimHits.CastorShowerLibrary.BranchEvt = 'hadShowerLibInfo.'
process.g4SimHits.CastorShowerLibrary.BranchEM  =  'emParticles.'
process.g4SimHits.CastorShowerLibrary.BranchHAD = 'hadParticles.'

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

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorTestAnalysis'),
    CastorTestAnalysis = cms.PSet(
        Verbosity           = cms.int32(0),
        EventNtupleFlag     = cms.int32(1),
        StepNtupleFlag      = cms.int32(0),
        EventNtupleFileName = cms.string('eventNtuple_pion_SL.root'),
        StepNtupleFileName  = cms.string('stepNtuple_pion_SL.root'),
    )
))



