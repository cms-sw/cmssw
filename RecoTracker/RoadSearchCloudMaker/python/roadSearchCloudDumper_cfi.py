import FWCore.ParameterSet.Config as cms

roadSearchCloudDumper = cms.EDAnalyzer("RoadSearchCloudDumper",
    RoadSearchCloudInputTag = cms.InputTag("roadSearchClouds"),
    RingsLabel = cms.string('')
)


