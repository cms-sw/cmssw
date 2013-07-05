import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalGeom', 
        'EcalSim', 
        'HCalGeom', 
        'HcalSim', 
        'HFShower'),
    destinations = cms.untracked.vstring('cout')
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckForHighEtPhotons = cms.untracked.bool(False),
    TrackMin     = cms.untracked.int32(0),
    TrackMax     = cms.untracked.int32(0),
    TrackStep    = cms.untracked.int32(1),
    EventMin     = cms.untracked.int32(0),
    EventMax     = cms.untracked.int32(0),
    EventStep    = cms.untracked.int32(1),
    PDGids       = cms.untracked.vint32(),
    VerboseLevel = cms.untracked.int32(0),
    G4Verbose = cms.untracked.bool(True),
    DEBUG        = cms.untracked.bool(False),
    type      = cms.string('TrackingVerboseAction')
))

