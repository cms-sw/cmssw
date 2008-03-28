import FWCore.ParameterSet.Config as cms

roadSearchClouds = cms.EDFilter("RoadSearchCloudMaker",
    # minimal fraction of user layers per cloud
    MinimalFractionOfUsedLayersPerCloud = cms.double(0.5),
    # module label of SiPixelRecHitConverter
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    # need true to reco field off cosmics
    StraightLineNoBeamSpotCloud = cms.bool(False),
    # Use Pixels or not
    UsePixelsinRS = cms.bool(True),
    # module label of RoadSearchSeedFinder
    SeedProducer = cms.InputTag("roadSearchSeeds"),
    # do cloud cleaning in CloudMaker instead of separate module
    DoCloudCleaning = cms.bool(True),
    # increase cuts in 0.9 < |eta| < 1.5 (transition region)
    IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud = cms.uint32(4),
    rphiStripRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    # Use stereo RecHits in addition to matched RecHits (double modules only)
    UseStereoRecHits = cms.bool(False),
    # Size of Z-Phi Road in Phi
    #double ZPhiRoadSize = 0.0020
    ZPhiRoadSize = cms.double(0.06),
    # maximal number of consecutive missed layers per road
    MaximalFractionOfConsecutiveMissedLayersPerCloud = cms.double(0.15),
    stereoStripRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    # minimal fraction of missed layers per cloud
    MaximalFractionOfMissedLayersPerCloud = cms.double(0.3),
    # scale factor for window around phi in which detid's are looked up within a ring
    scalefactorRoadSeedWindow = cms.double(1.5),
    # maximal number of RecHits per DetId in RoadSearchCloud
    MaxDetHitsInCloudPerDetId = cms.uint32(32),
    IncreaseMaxNumberOfMissedLayersPerCloud = cms.uint32(3),
    # roads service label
    RoadsLabel = cms.string(''),
    # maximal number of RecHits per RoadSearchCloud
    MaxRecHitsInCloud = cms.int32(100),
    # Use rphi RecHits in addition to matched RecHits (double modules only)
    UseRphiRecHits = cms.bool(False),
    # minimal fraction of hits which has to lap between RawRoadSearchClouds to be merged
    MergingFraction = cms.double(0.8),
    # Size of R-Phi Road in Phi
    RPhiRoadSize = cms.double(0.02),
    # strip rechit collections
    matchedStripRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    # minimum half road parameter
    MinimumHalfRoad = cms.double(0.55)
)


