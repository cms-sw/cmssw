import FWCore.ParameterSet.Config as cms

roadSearchEventFilter = cms.EDFilter("RoadSearchEventFilter",
    #untracked string SeedCollectionLabel = "roadSearchSeedsTIFTIBTOB"
    SeedCollectionLabel = cms.untracked.string('roadSearchSeedsTIF'),
    NumberOfSeeds = cms.untracked.uint32(1000)
)


