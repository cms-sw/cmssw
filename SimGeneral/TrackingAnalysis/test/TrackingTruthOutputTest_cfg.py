import FWCore.ParameterSet.Config as cms

process = cms.Process('TrackingTruthOutputTest')

process.load('FWCore/MessageService/MessageLogger_cfi')

process.load("SimGeneral.TrackingAnalysis.Playback_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source(
  'PoolSource',
  fileNames = cms.untracked.vstring('file:TrackingTruth.root')
)

process.myOutputTest = cms.EDAnalyzer(
  'TrackingTruthOutputTest',
  trackingTruth = cms.untracked.InputTag('mix', 'MergedTrackTruth'),
  # trackingTruth = cms.untracked.InputTag('mix'),
  dumpVertexes = cms.untracked.bool(False),
  dumpOnlyBremsstrahlung = cms.untracked.bool(False)  
)

process.p = cms.EndPath(process.mix*process.myOutputTest)
