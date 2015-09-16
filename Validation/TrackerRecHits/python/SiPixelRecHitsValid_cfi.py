import FWCore.ParameterSet.Config as cms

pixRecHitsValid = cms.EDAnalyzer("SiPixelRecHitsValid",
    src = cms.InputTag("siPixelRecHits"),
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof'),
    associateRecoTracks = cms.bool(False),
    associateStrip = cms.bool(False),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    verbose = cms.untracked.bool(False)
)


