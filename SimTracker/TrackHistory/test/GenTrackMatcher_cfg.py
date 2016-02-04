import FWCore.ParameterSet.Config as cms

process = cms.Process("MCCand")

process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimTracker.TrackHistory.GenTrackMatcher_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('')
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep *_genParticles_*_*', 
        'keep *_trackMCMatch_*_*'),
    fileName = cms.untracked.string('out.root')
)

process.p = cms.Path(process.genParticles*process.genTrackMatcher)
process.ep = cms.EndPath(process.out)
process.genParticles.saveBarCodes = True


