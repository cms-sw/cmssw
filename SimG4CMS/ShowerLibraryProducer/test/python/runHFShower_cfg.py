import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.HcalCommonData.hcalforwardshower_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('FiberSim', 
        'HFShower', 'HcalForwardLib'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FiberSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalForwardLib = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("FlatRandomEThetaGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID   = cms.untracked.vint32(11),
        MinTheta = cms.untracked.double(0.0),
        MaxTheta = cms.untracked.double(0.0),
        MinPhi   = cms.untracked.double(-3.1415926),
        MaxPhi   = cms.untracked.double(3.1415926),
        MinE     = cms.untracked.double(100.0),
        MaxE     = cms.untracked.double(100.0)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simevent.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ThfShowerLibSimu_100GeVElec.root')
)

process.p1 = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer")+process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.Generator.ApplyPCuts   = False
process.g4SimHits.Generator.ApplyEtaCuts = False
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


