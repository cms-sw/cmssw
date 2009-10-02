import FWCore.ParameterSet.Config as cms

process = cms.Process("FP420Test")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load('Configuration/StandardSequences/Generator_cff')

#process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/VtxSmearedGauss_cff')

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
# Input source
process.source = cms.Source("EmptySource")
process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(2212),
        MaxEta = cms.double(9.9),
#        MaxPhi = cms.double(3.227),
        MaxPhi = cms.double(3.1),
        MinEta = cms.double(8.7),
        MinE = cms.double(6930.0),
#        MinPhi = cms.double(3.053),
        MinPhi = cms.double(-3.1),
        MaxE = cms.double(7000.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single protons'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
                           )

ProductionFilterSequence = cms.Sequence(process.generator)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simfp420event.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
# first row of sensors
#process.VtxSmeared.MeanX = -1.0
# 2nd row of sensors
process.VtxSmeared.MeanX = -1.5
# 3rd row of sensors
#process.VtxSmeared.MeanX = -2.5
process.VtxSmeared.MeanY = 0.
process.VtxSmeared.MeanZ = 41900.0
process.VtxSmeared.SigmaX = 0.15
process.VtxSmeared.SigmaY = 0.15
process.VtxSmeared.SigmaZ = 1.0
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyPhiCuts = True
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    FP420Test = cms.PSet(
        Verbosity = cms.int32(1),
        FOutputFile = cms.string('TheAnlysis.root'),
        FDataLabel = cms.string('defaultData'),
        FRecreateFile = cms.string('RECREATE')
    ),
    type = cms.string('FP420Test')
))


