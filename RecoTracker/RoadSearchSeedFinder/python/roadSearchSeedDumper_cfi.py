import FWCore.ParameterSet.Config as cms

roadSearchSeedDumper = cms.EDFilter("RoadSearchSeedDumper",
    RingsLabel = cms.string(''),
    RoadSearchSeedInputTag = cms.InputTag("roadSearchSeeds")
)


