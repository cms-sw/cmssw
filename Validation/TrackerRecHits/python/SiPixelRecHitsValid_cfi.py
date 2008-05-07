import FWCore.ParameterSet.Config as cms

pixRecHitsValid = cms.EDFilter("SiPixelRecHitsValid",
    src = cms.InputTag("siPixelRecHits"),
    outputFile = cms.untracked.string('pixelrechitshisto.root'),
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof'),
    associateRecoTracks = cms.bool(False),
    associateStrip = cms.bool(False),
    verbose = cms.untracked.bool(True)
)


