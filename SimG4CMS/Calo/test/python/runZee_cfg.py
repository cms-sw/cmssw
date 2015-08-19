import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Calibration.IsolatedParticles.electronStudy_cfi")
process.load("FWCore.MessageService.MessageLogger_ReleaseValidation_cfi")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames =cms.untracked.vstring("file:zeer.root")
)

process.VtxSmeared = cms.EDProducer("EventVertexProducer")

from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import VertexSmearingParameters
process.VtxSmeared.VertexSmearing = cms.PSet(
    VertexSmearingParameters
)

process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('Simevent_zeer_QGSP_BERT_EML95.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('zeer_QGSP_BERT_EML95.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.electronStudy*process.rndmStore)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML95'

