import FWCore.ParameterSet.Config as cms

roadSearchSeedDumper = cms.EDAnalyzer("RoadSearchSeedDumper",
    RingsLabel = cms.string(''),
    RoadSearchSeedInputTag = cms.InputTag("roadSearchSeeds")
)


