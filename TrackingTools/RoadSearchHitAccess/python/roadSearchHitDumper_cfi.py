import FWCore.ParameterSet.Config as cms

roadSearchHitDumper = cms.EDFilter("RoadSearchHitDumper",
    stereoStripRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    rphiStripRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    # rings service label
    RingsLabel = cms.string(''),
    # strip rechit collections
    matchedStripRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    # module label of SiPixelRecHitConverter
    pixelRecHits = cms.InputTag("siPixelRecHits")
)


