import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedBeamProfile_cfi")
process.load("SimG4Core.GFlash.TB2006GeometryXML_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.common_beam_direction_parameters = cms.PSet(
    MaxEta = cms.untracked.double(0.2175),
    MaxPhi = cms.untracked.double(-0.1309),
    MinEta = cms.untracked.double(0.2175),
    MinPhi = cms.untracked.double(-0.1309),
    BeamPosition = cms.untracked.double(-800.0)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        MaxE = cms.untracked.double(20.0001),
        MinE = cms.untracked.double(20.0),
        PartID = cms.untracked.vint32(-211)
    ),
    Verbosity = cms.untracked.int32(0),
    firstEvent = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("SimG4Core.Application.g4SimHits_cfi")
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
process.g4SimHits.Physics.GFlash = cms.PSet(
    GflashHadronPhysics = cms.string('QGSP_BERT'),
    GflashEMShowerModel = cms.bool(True),
    GflashHadronShowerModel = cms.bool(True),
    GflashHistogram = cms.bool(True),
    GflashHistogramName = cms.string('gflash_histogram_h2.root'),
    bField = cms.double(0.0),
    tuning_pList = cms.vdouble()
)
process.g4SimHits.CaloSD.BeamPosition = cms.untracked.double(-800)
process.g4SimHits.HCalSD.ForTBH2 = cms.untracked.bool(True)
process.g4SimHits.HCalSD.UseHF = cms.untracked.bool(False)
process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(False)
process.g4SimHits.CaloTrkProcessing.TestBeam = True

process.Timing = cms.Service("Timing")

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('gflash_tbH2.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

