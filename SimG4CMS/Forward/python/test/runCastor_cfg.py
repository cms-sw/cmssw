import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.StandardSequences.GeometryExtended_cff")
#process.load("SimG4CMS.Forward.castorGeometryXML_cfi")
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.RandomNumberGeneratorService.theSource.initialSeed = 113456789
process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)


process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MaxEta = cms.double(5.9),
        MaxPhi = cms.double(0.0),
        MinEta = cms.double(5.9),
        MinE = cms.double(50.00),
        MinPhi = cms.double(0.392),
        MaxE = cms.double(50.01)
    ),
    AddAntiParticle = cms.bool(True),
    Verbosity = cms.untracked.int32(0)
)

process.CaloSD = cms.PSet(
    DetailedTiming = cms.bool(False),
    EminTrack = cms.double(1.0),
    Verbosity = cms.int32(0),
    UseMap = cms.bool(True),
    CheckHits = cms.int32(25)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sim_muon.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 10.
process.g4SimHits.Generator.MinEtaCut = -7.0
process.g4SimHits.Generator.MaxEtaCut = 7.0
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorTestAnalysis'),
    CastorTestAnalysis = cms.PSet(
        EventNtupleFileName = cms.string('eventNtuple_muon.root'),
        Verbosity = cms.int32(0),
        StepNtupleFileName = cms.string('stepNtuple.root'),
        StepNtupleFlag = cms.int32(0),
        EventNtupleFlag = cms.int32(1)
    )
))

