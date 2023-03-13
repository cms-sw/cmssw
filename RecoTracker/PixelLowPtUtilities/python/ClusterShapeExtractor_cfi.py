import FWCore.ParameterSet.Config as cms

clusterShapeExtractor = cms.EDAnalyzer("PixelClusterShapeExtractor",
    tracks = cms.InputTag("generalTracks"),
    clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    pixelSimLinkSrc = cms.InputTag('simSiPixelDigis'),
    hasSimHits     = cms.bool(True),
    hasRecTracks   = cms.bool(False),
    noBPIX1   = cms.bool(False),
# for the associator
    associateStrip      = cms.bool(False),
    associatePixel      = cms.bool(True),
    associateRecoTracks = cms.bool(False),
    ROUList = cms.vstring(
      'TrackerHitsPixelBarrelLowTof',
      'TrackerHitsPixelBarrelHighTof',
      'TrackerHitsPixelEndcapLowTof',
      'TrackerHitsPixelEndcapHighTof')
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(clusterShapeExtractor,
    pixelSimLinkSrc = 'simSiPixelDigis:Pixel'
)
