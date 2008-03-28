import FWCore.ParameterSet.Config as cms

#
# standard parameter-set entries for module
#
# RoadSeachCloudCleaner
#
# located in
#
# RecoTracker/RoadSearchCloudCleaner
#
# 
# sequence dependency:
#
# - RawRoadSearchClouds: include "RecoTracker/RoadSearchCloudMaker/data/RoadSearchCloudMaker.cfi
#
#
# service dependency:
#
#
# function:
#
# merges RoadSearchClouds according to hit overlap
cleanRoadSearchClouds = cms.EDFilter("RoadSearchCloudCleaner",
    # maximal number of RecHits per RoadSearchCloud
    MaxRecHitsInCloud = cms.int32(100),
    # minimal fraction of hits which has to lap between RawRoadSearchClouds to be merged
    MergingFraction = cms.double(0.7),
    # module label of RoadSearchCloudMaker
    RawCloudProducer = cms.string('rawRoadSearchClouds')
)


