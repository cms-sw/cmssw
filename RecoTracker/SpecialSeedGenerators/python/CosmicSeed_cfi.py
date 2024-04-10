import FWCore.ParameterSet.Config as cms

cosmicseedfinder = cms.EDProducer("CosmicSeedGenerator",
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'), ##other choice: TIBD+

    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MaxNumberOfStripClusters = cms.uint32(300),
    maxSeeds = cms.int32(10000),
    SeedPt = cms.double(5.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.9),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    doClusterCheck = cms.bool(True),
    DontCountDetsAboveNClusters = cms.uint32(20),
    originRadius = cms.double(150.0),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    originHalfLength = cms.double(90.0),
    #***top-bottom
    PositiveYOnly = cms.bool(False),
    NegativeYOnly = cms.bool(False)  
    #***
)


