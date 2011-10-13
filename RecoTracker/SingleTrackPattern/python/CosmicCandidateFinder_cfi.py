import FWCore.ParameterSet.Config as cms

cosmicCandidateFinder = cms.EDProducer("CosmicTrackFinder",
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    HitProducer = cms.string('siStripRecHits'),
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MinHits = cms.int32(4),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    debug = cms.untracked.bool(True),
    GeometricStructure = cms.untracked.string('MTCC'),
    cosmicSeeds = cms.InputTag("cosmicseedfinder")
)


