import FWCore.ParameterSet.Config as cms

gemRecHitsValidation = cms.EDAnalyzer("MuonGEMRecHits",
    debug = cms.untracked.bool(False),
    gemRecHitInput = cms.untracked.InputTag('gemRecHits'),
    gemSimHitInput = cms.untracked.InputTag('g4SimHits','MuonGEMHits'),
    simTrackInput = cms.untracked.InputTag('g4SimHits'),
    folderPath = cms.untracked.string('MuonGEMRecHitsV/GEMRecHitTask'),
    outputFile = cms.untracked.string('GEMRecHit_ME.root')
)
