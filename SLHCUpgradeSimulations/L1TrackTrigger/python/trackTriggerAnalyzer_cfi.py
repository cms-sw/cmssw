
import FWCore.ParameterSet.Config as cms

trackTriggerAnalyzer = cms.EDProducer("TrackTriggerAnalyzer",
  hitsTag = cms.InputTag("trackTriggerHits"),
  stubsTag = cms.InputTag("none"),
  nLayers = cms.uint32(1),
  nBinsIZ = cms.uint32(100),
  nBinsIPhi = cms.uint32(100),
)                                     


