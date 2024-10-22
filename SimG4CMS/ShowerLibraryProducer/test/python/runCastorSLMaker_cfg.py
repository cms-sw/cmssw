import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorShowerLibraryMaker")

process.common_maximum_timex = cms.PSet(
  MaxTrackTime = cms.double(500.0),
  MaxTimeNames = cms.vstring(),
  MaxTrackTimes = cms.vdouble()
)

process.common_pgun_particleID = cms.PSet(
        PartID = cms.vint32(11,211)
        #PartID = cms.vint32(211)
)

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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
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
    input = cms.untracked.int32(1000000000)
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
    CheckHits      = cms.int32(25),
    TmaxHit        = cms.int32(500)  # L.M. testing
)

process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   process.common_maximum_timex,        # need to be localy redefined
   TrackNeutrino = cms.bool(False),
   KillHeavy     = cms.bool(False),
   KillDeltaRay  = cms.bool(False),
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

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_pgun_particleID,
        MinEta = cms.double(-6.6),
        MaxEta = cms.double(-5.2),
        MinPhi = cms.double(0.),
        MaxPhi = cms.double(0.7854), # PI/4 = 0.7854
        MinE = cms.double(12.00),
        #MeanE = cms.double(12.00),
        MaxE = cms.double(14.00)
        #Energybins = cms.vdouble(1.,2.,3.,5.,7.,10.,20.,30.,45.,60.,75.,100.,140.,200.,300.,600.,1000.,1500.)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(False)
)

process.g4SimHits.CastorSD.useShowerLibrary = False

process.source = cms.Source("EmptySource")
#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('sim_pion_1events-ppON.root')
#)


process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorShowerLibraryMaker'),
    CastorShowerLibraryMaker = cms.PSet(
        process.common_pgun_particleID,
        EventNtupleFileName = cms.string('SL_had_E12GeV_eta-6.0phi0.3_1events-ppON.root'),
        Verbosity = cms.int32(0),
        DeActivatePhysicsProcess = cms.bool(False),
        StepNtupleFileName = cms.string('stepNtuple_pion_electron_E12GeV_1event-ppON.root'),
        StepNtupleFlag = cms.int32(0),
        EventNtupleFlag = cms.int32(0),
        # for shower library
        nemEvents       = cms.int32(5),
        SLemEnergyBins  = cms.vdouble(10.),
        SLemEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLemPhiBins     = cms.vdouble(0.,0.07854,0.15708,0.23562,0.31416,0.3927,0.47124,0.54978,0.62832,0.70686),
        nhadEvents       = cms.int32(5),
        SLhadEnergyBins  = cms.vdouble(10.),
        #SLhadEnergyBins  = cms.vdouble(1.,2.,3.,5.,7.,10.,20.,30.,45.,60.,75.,100.,140.,200.),
        SLhadEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLhadPhiBins     = cms.vdouble(0.,0.07854,0.15708,0.23562,0.31416,0.3927,0.47124,0.54978,0.62832,0.70686),
        SLMaxPhi         = cms.double(0.7854),
        SLMaxEta         = cms.double(-5.2)
    )
))


process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
#process.outpath = cms.EndPath(process.o1)
