import FWCore.ParameterSet.Config as cms

globalCombinedSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag( 
        cms.InputTag("globalSeedsFromTripletsWithVertices"),
        cms.InputTag("globalSeedsFromPairsWithVertices"),
    )
    # the following allows to re-key the cluster reference of the rechits on the seeds that are put together
    #N.B the cluster removal infos should be the same as the one use in the corresponding track producer.
    #,clusterRemovalInfos = cms.VInputTag(cms.InputTag(""),cms.InputTag("clusterRemovalForGlobalSeedsFromPairsWithVertices"))
       
)


