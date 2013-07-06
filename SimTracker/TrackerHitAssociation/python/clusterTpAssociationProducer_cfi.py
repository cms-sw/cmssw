import FWCore.ParameterSet.Config as cms

tpClusterProducer = cms.EDProducer("ClusterTPAssociationProducer",
  verbose = cms.bool(False),
  simTrackSrc     = cms.InputTag("g4SimHits"),
  pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
  stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
  pixelClusterSrc = cms.InputTag("siPixelClusters"),
  stripClusterSrc = cms.InputTag("siStripClusters"),
  trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth')
)
