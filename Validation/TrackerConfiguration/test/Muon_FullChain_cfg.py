import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimG4Core.Configuration.SimG4Core_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Validation.TrackerHits.trackerHitsValidation_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimTracker.Configuration.SimTracker_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")

process.load("Validation.RecoTrack.TrackValidation_cff")

process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:./Muon.root')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Muon_FullValidation.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits*process.trackerHitsValidation)
process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.rechits = cms.Sequence(process.trackerlocalreco*process.trackerRecHitsValidation)
process.tracks = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.trackingParticles*process.trackingTruthValid*process.ckftracks*process.trackerRecHitsValidation)
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.rechits*process.tracks*process.trackinghits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'


