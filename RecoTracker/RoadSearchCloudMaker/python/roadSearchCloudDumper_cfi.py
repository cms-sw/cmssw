import FWCore.ParameterSet.Config as cms

roadSearchCloudDumper = cms.EDFilter("RoadSearchCloudDumper",
    RoadSearchCloudInputTag = cms.InputTag("roadSearchClouds"),
    RingsLabel = cms.string('')
)


