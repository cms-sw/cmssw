import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.Timing = cms.Service("Timing")

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.generator = cms.EDProducer("FlatRandomMultiParticlePGunProducer",
    PGunParameters = cms.PSet(
        PartID    = cms.vint32(211,321,2212),
        ProbParts = cms.vdouble(0.695,0.163,0.142),
        MinEta    = cms.double(-2.3),
        MaxEta    = cms.double(2.3),
        MinPhi    = cms.double(-3.1415926),
        MaxPhi    = cms.double(3.1415926),
        MinP      = cms.double(1.0),
        MaxP      = cms.double(20.5)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True),
    firstRun        = cms.untracked.uint32(1)
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.FileNameGDML = 'cms2016.gdml'

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

