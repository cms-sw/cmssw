import FWCore.ParameterSet.Config as cms

siStripClusterThinningProducer = cms.EDProducer("SiStripClusterThinningProducer",
                                                inputTag = cms.InputTag("siStripClusters"),
                                                trackingRecHitsTags = cms.VInputTag(),
                                                )
