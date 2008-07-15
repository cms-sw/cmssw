import FWCore.ParameterSet.Config as cms

# Analyze and plot the tracking material
trackingMaterialAnalyser = cms.EDFilter("TrackingMaterialAnalyser",
    MaterialAccounting      = cms.InputTag("trackingMaterialProducer"),
    SplitMode               = cms.string("NearestLayer"),
    SkipBeforeFirstDetector = cms.bool(False),
    SkipAfterLastDetector   = cms.bool(True),
    SymmetricForwardLayers  = cms.bool(False),
    SaveSummaryPlot         = cms.bool(True),
    SaveDetailedPlots       = cms.bool(False),
    SaveParameters          = cms.bool(True)
)
