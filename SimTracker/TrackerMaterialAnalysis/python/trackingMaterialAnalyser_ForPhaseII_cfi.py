import FWCore.ParameterSet.Config as cms

trackingMaterialAnalyser = cms.EDAnalyzer("TrackingMaterialAnalyser",
    MaterialAccounting      = cms.InputTag("trackingMaterialProducer"),
    SplitMode               = cms.string("NearestLayer"),
    SkipBeforeFirstDetector = cms.bool(False),
    SkipAfterLastDetector   = cms.bool(False),
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
        "TrackerRecMaterialPhase2PixelEndcapDisk1Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk2Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk3Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk4Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk5Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk6Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk7Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk8Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk9Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk10Fw",
        "TrackerRecMaterialPhase2PixelEndcapDisk11Fw",
        "TrackerRecMaterialTOBPixelBarrelLayer1",
        "TrackerRecMaterialTOBPixelBarrelLayer2",
        "TrackerRecMaterialTOBPixelBarrelLayer3",
        "TrackerRecMaterialTOBPixelBarrelLayer4",
        "TrackerRecMaterialTOBPixelBarrelLayer5",
        "TrackerRecMaterialTOBPixelBarrelLayer6",
        "TrackerRecMaterialTIDDisk3",
        "TrackerRecMaterialTIDDisk4",
        "TrackerRecMaterialTIDDisk5"
    )
)
