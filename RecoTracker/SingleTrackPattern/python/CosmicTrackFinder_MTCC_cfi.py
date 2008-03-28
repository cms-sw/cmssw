import FWCore.ParameterSet.Config as cms

cosmictrackfinder = cms.EDFilter("CosmicTrackFinder",
    TrajInEvents = cms.bool(True),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    HitProducer = cms.string('siStripRecHits'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MinHits = cms.int32(3),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    GeometricStructure = cms.untracked.string('MTCC'),
    cosmicSeeds = cms.InputTag("cosmicseedfinder")
)


