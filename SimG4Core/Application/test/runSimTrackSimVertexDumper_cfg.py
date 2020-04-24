import FWCore.ParameterSet.Config as cms

process = cms.Process("SimTkVtxDump")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myfile.root')
)

process.prod = cms.EDAnalyzer("SimTrackSimVertexDumper",
    moduleLabelTk = cms.InputTag('g4SimHits'),
    moduleLabelVtx = cms.InputTag('g4SimHits'),
    dumpHepMC = cms.untracked.bool(True),
    moduleLabelHepMC = cms.InputTag('VtxSmeared')
)

process.p1 = cms.Path(process.prod)


