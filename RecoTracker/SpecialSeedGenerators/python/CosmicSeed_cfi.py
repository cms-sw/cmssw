import FWCore.ParameterSet.Config as cms

cosmicseedfinder = cms.EDFilter("CosmicSeedGenerator",
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    SeedPt = cms.double(1.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.9),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    originRadius = cms.double(150.0),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit")
)


