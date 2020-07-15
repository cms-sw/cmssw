import FWCore.ParameterSet.Config as cms

siPixelClusterThinningProducer = cms.EDProducer("SiPixelClusterThinningProducer",
                                                inputTag = cms.InputTag("siPixelClusters"),
                                                trackingRecHitsTags = cms.VInputTag(),
                                                )
