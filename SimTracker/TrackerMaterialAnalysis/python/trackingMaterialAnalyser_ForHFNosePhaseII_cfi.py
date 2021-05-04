import FWCore.ParameterSet.Config as cms

trackingMaterialAnalyser = cms.EDAnalyzer("TrackingMaterialAnalyser",
    MaterialAccounting      = cms.InputTag("trackingMaterialProducer"),
    SplitMode               = cms.string("NearestLayer"),
    SkipBeforeFirstDetector = cms.bool(False),
    SkipAfterLastDetector   = cms.bool(True),
    SaveSummaryPlot         = cms.bool(True),
    SaveDetailedPlots       = cms.bool(False),
    SaveParameters          = cms.bool(True),
    SaveXML                 = cms.bool(True),
    isHGCal                 = cms.bool(False),
    isHFNose                = cms.bool(True),
    Groups = cms.vstring(
        "HGCalHFNoseSensitive"
    )
)
