import FWCore.ParameterSet.Config as cms

L1TkClustersFromSimHits = cms.EDProducer("L1TkClusterBuilder_PSimHit_",
    rawHits = cms.VInputTag(cms.InputTag("g4SimHits","TrackerHits")),
    simTrackHits = cms.InputTag("g4SimHits"),
)

L1TkClustersFromPixelDigis = cms.EDProducer("L1TkClusterBuilder_PixelDigi_",
    rawHits = cms.VInputTag(cms.InputTag("simSiPixelDigis")),
    simTrackHits = cms.InputTag("g4SimHits"),
    ADCThreshold = cms.uint32(30),
)

