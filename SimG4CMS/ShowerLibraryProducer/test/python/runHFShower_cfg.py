import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.HcalCommonData.hcalforwardshower_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FiberSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalForwardLib = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('FiberSim', 
        'G4cout', 
        'G4cerr', 
        'HFShower', 
        'HcalForwardLib'),
    destinations = cms.untracked.vstring('cout')
)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(11),
        MaxEta = cms.untracked.double(5.5),
        MaxPhi = cms.untracked.double(3.1415926),
        MinEta = cms.untracked.double(5.4),
        MinE = cms.untracked.double(10.0),
        MinPhi = cms.untracked.double(-3.1415926),
        MaxE = cms.untracked.double(10.0)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simevent2.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ThfShowerLibSimu_100GeVElec.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.Physics.DefaultCutValue = 0.1
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HFShowerLibraryProducer = cms.PSet(
        Names = cms.vstring('FibreHits', 
            'ChamberHits', 
            'WedgeHits')
    ),
    type = cms.string('HcalForwardAnalysis')
))


