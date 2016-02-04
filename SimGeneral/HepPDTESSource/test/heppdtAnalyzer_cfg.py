import FWCore.ParameterSet.Config as cms

process = cms.Process("heppdtAnalyzer")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.dummy = cms.EDAnalyzer("HepPDTAnalyzer",
    particleName = cms.string("all")
)

process.p = cms.Path(process.dummy)


