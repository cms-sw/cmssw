import FWCore.ParameterSet.Config as cms

stripRecHitsValid = cms.EDFilter("SiStripRecHitsValid",
    outputFile = cms.untracked.string('sistriprechitshisto.root'),
    associatePixel = cms.bool(False),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 'TrackerHitsTIBHighTof', 'TrackerHitsTIDLowTof', 'TrackerHitsTIDHighTof', 'TrackerHitsTOBLowTof', 'TrackerHitsTOBHighTof', 'TrackerHitsTECLowTof', 'TrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(False),
    associateStrip = cms.bool(True),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    RecHitProducer = cms.string('siStripMatchedRecHits'),
    verbose = cms.untracked.bool(True)
)


