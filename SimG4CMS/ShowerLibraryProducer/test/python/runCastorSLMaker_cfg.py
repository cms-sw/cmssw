import FWCore.ParameterSet.Config as cms

common_maximum_timex = cms.PSet(
  MaxTrackTime = cms.double(500.0),
  MaxTimeNames = cms.vstring(),
  MaxTrackTimes = cms.vdouble()
)

common_pgun_particleID = cms.PSet(
        PartID = cms.vint32(11,211)
        #PartID = cms.vint32(11)
)

process = cms.Process("CastorShowerLibraryMaker")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("SimG4CMS.Forward.castorGeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

#process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
#    categories = cms.untracked.vstring('ForwardSim'),
#    debugModules = cms.untracked.vstring('*'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
#        DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        ForwardSim = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        )
#    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits  = cms.untracked.uint32(8245),         # std: 9784
        VtxSmeared = cms.untracked.uint32(123456789),
        generator  = cms.untracked.uint32(536870912)     # std: 135799753
    )
    #sourceSeed = cms.untracked.uint32(135799753)         # std: 135799753
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)

process.source = cms.Source("EmptySource")
process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_pion.root')
)


process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 10.
process.g4SimHits.Generator.MinEtaCut        = -7.0
process.g4SimHits.Generator.MaxEtaCut        = 7.0
process.g4SimHits.Generator.Verbosity        = 0
process.g4SimHits.CaloTrkProcessing.TestBeam = True

process.CaloSD = cms.PSet(
    DetailedTiming = cms.bool(False),
    EminTrack      = cms.double(1.0),
    Verbosity      = cms.int32(0),
    UseMap         = cms.bool(True),
    CheckHits      = cms.int32(25)
)

process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   common_maximum_timex,        # need to be localy redefined
   TrackNeutrino = cms.bool(False),
   KillHeavy     = cms.bool(False),
   SaveFirstLevelSecondary = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)

process.g4SimHits.SteppingAction = cms.PSet(
   common_maximum_timex, # need to be localy redefined
   KillBeamPipe            = cms.bool(True),
   CriticalEnergyForVacuum = cms.double(2.0),
   CriticalDensity         = cms.double(1e-15),
   EkinNames               = cms.vstring(),
   EkinThresholds          = cms.vdouble(),
   EkinParticles           = cms.vstring(),
   Verbosity               = cms.untracked.int32(0)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        common_pgun_particleID,
        MinEta = cms.double(-6.6),
        MaxEta = cms.double(-5.2),
        MinPhi = cms.double(0.0),
        MaxPhi = cms.double(0.7854),
        MinE = cms.double(1.00),
        MaxE = cms.double(100.00)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(False)
)


process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorShowerLibraryMaker'),
    CastorShowerLibraryMaker = cms.PSet(
        common_pgun_particleID,
        EventNtupleFileName = cms.string('SL_em+had_E1:5:10:30:60:100GeV_7eta-6.6:-5.2_10phi0:0.7854_10events.root'),
        Verbosity = cms.int32(0),
        StepNtupleFileName = cms.string('stepNtuple_pion_electron.root'),
        StepNtupleFlag = cms.int32(0),
        EventNtupleFlag = cms.int32(1),
        # for shower library
        nemEvents       = cms.int32(10),
        SLemEnergyBins  = cms.vdouble(1.0,5.0,10.,30.,60.),
        SLemEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLemPhiBins     = cms.vdouble(0.,0.07854,0.15708,0.23562,0.31416,0.3927,0.47124,0.54978,0.62832,0.70686),
#        nhadEvents       = cms.int32(0),
#        SLhadEnergyBins  = cms.vdouble(),
#        SLhadEtaBins     = cms.vdouble(),
#        SLhadPhiBins     = cms.vdouble()
        nhadEvents       = cms.int32(10),
        SLhadEnergyBins  = cms.vdouble(1.0,5.0,10.,30.,60.),
        SLhadEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLhadPhiBins     = cms.vdouble(0.,0.07854,0.15708,0.23562,0.31416,0.3927,0.47124,0.54978,0.62832,0.70686)
    )
))

process.g4SimHits.CastorSD.useShowerLibrary = False

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
