import FWCore.ParameterSet.Config as cms

process = cms.Process("SimTkVtxDump")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.load("SimG4Core.Application.simTrackSimVertexDumper_cfi")
process.simTrackSimVertexDumper.dumpHepMC = True

process.p1 = cms.Path(process.simTrackSimVertexDumper)
