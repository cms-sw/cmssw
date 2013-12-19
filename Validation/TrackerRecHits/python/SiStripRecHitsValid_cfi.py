import FWCore.ParameterSet.Config as cms

stripRecHitsValid = cms.EDAnalyzer("SiStripRecHitsValid",
    outputFile = cms.untracked.string(''),
    associatePixel = cms.bool(False),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof', 
        'g4SimHitsTrackerHitsTIBHighTof', 
        'g4SimHitsTrackerHitsTIDLowTof', 
        'g4SimHitsTrackerHitsTIDHighTof', 
        'g4SimHitsTrackerHitsTOBLowTof', 
        'g4SimHitsTrackerHitsTOBHighTof', 
        'g4SimHitsTrackerHitsTECLowTof', 
        'g4SimHitsTrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(False),
    associateStrip = cms.bool(True),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    RecHitProducer = cms.string('siStripMatchedRecHits'),
    verbose = cms.untracked.bool(False)
)


