import FWCore.ParameterSet.Config as cms

globalCombinedSeeds = cms.EDFilter("SeedCombiner",
    TripletCollection = cms.InputTag("globalSeedsFromTripletsWithVertices"),
    PairCollection = cms.InputTag("globalSeedsFromPairsWithVertices")
)


