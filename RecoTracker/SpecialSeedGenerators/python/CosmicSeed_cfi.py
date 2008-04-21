import FWCore.ParameterSet.Config as cms

cosmicseedfinder = cms.EDFilter("CosmicSeedGenerator",
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'), ##other choice: TIBD+

    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    SeedPt = cms.double(1.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.9),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    doClusterCheck = cms.bool(True),
    originRadius = cms.double(150.0),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    originHalfLength = cms.double(90.0)
)


