import FWCore.ParameterSet.Config as cms

#
# standard parameter-set entries for module
#
# RoadSeachSeedFinder
#
# located in
#
# RecoTracker/RoadSearchSeedFinder
#
# function:
#
# produces RoadSearchSeeds for seeding track reconstruction 
#
roadSearchSeeds = cms.EDFilter("RoadSearchSeedFinder",
    # TrackingRecHit access configuration for outer seed rings
    # access mode for TrackingTools/RoadSearchHitAccess, allowed values: "STANDARD",'RPHI'
    OuterSeedRecHitAccessMode = cms.string('RPHI'),
    # module label of SiPixelRecHitConverter
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    MaximalEndcapImpactParameter = cms.double(1.2),
    MergeSeedsCenterCut_C = cms.double(0.4),
    MergeSeedsCenterCut_B = cms.double(0.25),
    # seed cleaning cuts, if two compared seeds fulfill cuts, newer is dropped
    # cuts on percentage of difference in seed circle centers vs. average of seed circle centers and
    # percentage of difference in seed circle radii vs. average of seed circle radii
    # in three different eta regions: |eta| <= 1.1 (A), 1.1 < |eta| <= 1.6 (B), |eta| > 1.6 (C)
    MergeSeedsCenterCut_A = cms.double(0.05),
    # cut on maximal number of hits different between circle seeds to merge, all hits have to be on different layers
    MergeSeedsDifferentHitsCut = cms.uint32(1),
    rphiStripRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    # maximal impact parameter cut on seeds in cm
    MaximalBarrelImpactParameter = cms.double(0.2),
    # restrict RS cosmic reconstruction to events which have less than MaxNumberOfClusters
    MaxNumberOfClusters = cms.uint32(300),
    stereoStripRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    # roads service label
    RoadsLabel = cms.string(''),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    # In the case of double sided sensors, return in addition to matched also stereo rechits which have not been matched
    OuterSeedRecHitAccessUseStereo = cms.bool(False),
    # minimal transverse momentum of reconstructed tracks cut on seeds in GeV
    MinimalReconstructedTransverseMomentum = cms.double(1.5),
    # phi range in radians to restrict loop over detid's in rings
    PhiRangeForDetIdLookupInRings = cms.double(0.5),
    # seeding mode: STANDARD, STRAIGHT-LINE
    Mode = cms.string('STANDARD'),
    # enable/disable cosmic tracking mode
    CosmicTracking = cms.bool(False),
    MergeSeedsRadiusCut_A = cms.double(0.05),
    # TrackingRecHit access configuration for inner seed rings
    # access mode for TrackingTools/RoadSearchHitAccess, allowed values: "STANDARD",'RPHI'
    InnerSeedRecHitAccessMode = cms.string('RPHI'),
    # In the case of double sided sensors, return in addition to matched also stereo rechits which have not been matched
    InnerSeedRecHitAccessUseStereo = cms.bool(False),
    # In the case of double sided sensors, return in addition to matched also rphi rechits which have not been matched
    OuterSeedRecHitAccessUseRPhi = cms.bool(False),
    MergeSeedsRadiusCut_B = cms.double(0.25),
    MergeSeedsRadiusCut_C = cms.double(0.4),
    # strip rechit collections
    matchedStripRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    # In the case of double sided sensors, return in addition to matched also rphi rechits which have not been matched
    InnerSeedRecHitAccessUseRPhi = cms.bool(False)
)


