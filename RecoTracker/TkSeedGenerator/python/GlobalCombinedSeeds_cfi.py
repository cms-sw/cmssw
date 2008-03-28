import FWCore.ParameterSet.Config as cms

globalCombinedSeeds = cms.EDFilter("SeedCombiner",
    TripletCollection = cms.untracked.string('globalSeedsFromTripletsWithVertices'),
    PairCollection = cms.untracked.string('globalSeedsFromPairsWithVertices')
)


