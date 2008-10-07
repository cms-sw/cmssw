import FWCore.ParameterSet.Config as cms

globalCombinedSeeds = cms.EDFilter("SeedCombiner",
    seedCollections = cms.VInputTag( 
        cms.InputTag("globalSeedsFromTripletsWithVertices"),
        cms.InputTag("globalSeedsFromPairsWithVertices"),
    )
)


