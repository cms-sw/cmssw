import FWCore.ParameterSet.Config as cms

# Analyze and plot the tracking material
trackingMaterialAnalyser = cms.EDFilter("TrackingMaterialAnalyser",
    SymmetricForwardLayers = cms.bool(False),
    MaterialAccounting = cms.InputTag("trackingMaterialProducer"),
    SkipAfterLastDetector = cms.bool(True),
    SkipBeforeFirstDetector = cms.bool(False)
)


