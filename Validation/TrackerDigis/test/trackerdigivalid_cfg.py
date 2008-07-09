import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("SimTracker.Configuration.SimTracker_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/genta/686C53F9-E933-DD11-B74E-001617DC1F70.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.p1 = cms.Path(process.mix*process.digis)


