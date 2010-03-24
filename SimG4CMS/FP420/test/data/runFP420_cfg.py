import FWCore.ParameterSet.Config as cms

process = cms.Process("FP420Test")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(2212),
        MaxEta = cms.untracked.double(9.9),
#        MaxPhi = cms.untracked.double(3.227),
        MaxPhi = cms.untracked.double(3.1),
        MinEta = cms.untracked.double(8.7),
        MinE = cms.untracked.double(6930.0),
#        MinPhi = cms.untracked.double(3.053),
        MinPhi = cms.untracked.double(-3.1),
        MaxE = cms.untracked.double(7000.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simfp420event.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
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


