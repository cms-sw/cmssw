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
# to derive the following list:
# cat ../data/trackingMaterialGroups_ForPhaseII.xml | grep TrackingMaterialGroup | sed -e 's/\s*//' | cut -d ' ' -f 3 | tr '=' ' ' | cut -d ' ' -f 2 | tr -d '"' | sed -e 's/\(.*\)/"\1",/'
    Groups = cms.vstring(
        "TrackerRecMaterialPhase1PixelBarrelLayer1",
        "TrackerRecMaterialPhase1PixelBarrelLayer2",
        "TrackerRecMaterialPhase1PixelBarrelLayer3",
        "TrackerRecMaterialPhase1PixelBarrelLayer4",
        "TrackerRecMaterialPhase2PixelForwardDisk1",
        "TrackerRecMaterialPhase2PixelForwardDisk2",
        "TrackerRecMaterialPhase2PixelForwardDisk3",
        "TrackerRecMaterialPhase2PixelForwardDisk4",
        "TrackerRecMaterialPhase2PixelForwardDisk5",
        "TrackerRecMaterialPhase2PixelForwardDisk6",
        "TrackerRecMaterialPhase2PixelForwardDisk7",
        "TrackerRecMaterialPhase2PixelForwardDisk8",
        "TrackerRecMaterialPhase2PixelForwardDisk9",
        "TrackerRecMaterialPhase2PixelForwardDisk10",
        "TrackerRecMaterialPhase2PixelForwardDisk11",
        "TrackerRecMaterialPhase2OTBarrelLayer1",
        "TrackerRecMaterialPhase2OTBarrelLayer2",
        "TrackerRecMaterialPhase2OTBarrelLayer3",
        "TrackerRecMaterialPhase2OTBarrelLayer4",
        "TrackerRecMaterialPhase2OTBarrelLayer5",
        "TrackerRecMaterialPhase2OTBarrelLayer6",
        "TrackerRecMaterialPhase2OTForwardDisk1",
        "TrackerRecMaterialPhase2OTForwardDisk2",
        "TrackerRecMaterialPhase2OTForwardDisk3",
        "TrackerRecMaterialPhase2OTForwardDisk4",
        "TrackerRecMaterialPhase2OTForwardDisk5"
    )
)
