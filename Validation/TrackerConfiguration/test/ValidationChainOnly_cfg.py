import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#process.load("Validation.TrackerConfiguration.RelValSingleMuPt10_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("SimTracker.Configuration.SimTracker_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Validation.TrackerHits.trackerHitsValidation_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")

process.load("Validation.RecoTrack.TrackValidation_cff")

process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleMuPt10-1214048167-IDEAL_V2-2nd/0004/0AE2B3E3-0141-DD11-846F-000423D98BC4.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.trackerHitsValidation)
process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.rechits = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*process.trackerRecHitsValidation)
process.tracks = cms.Sequence(process.trackingTruthValid*process.tracksValidation)
process.trackinghits = cms.Sequence(process.trackingRecHitsValid)
process.p1 = cms.Path(process.mix*process.simhits*process.digis*process.rechits*process.tracks*process.trackinghits)
