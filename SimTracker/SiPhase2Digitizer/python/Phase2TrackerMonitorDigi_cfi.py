import FWCore.ParameterSet.Config as cms

digiMon = cms.EDAnalyzer("Phase2TrackerMonitorDigi",
    Verbosity = cms.bool(False),
    TopFolderName = cms.string("Phase2Tracker"),
    PixelDigiSource    = cms.InputTag("simSiPixelDigis","Pixel"),                          
    OuterTrackerDigiSource    = cms.InputTag("mix", "Tracker"),                          
    NumbeOfDigisH = cms.PSet(
           Nbins = cms.int32(200),
           xmin = cms.double(-0.5),
           xmax = cms.double(200.5)
    ),
    PositionOfDigisH = cms.PSet(
           Nxbins = cms.int32(260),
           xmin   = cms.double(0.5),
           xmax   = cms.double(260.5),
           Nybins = cms.int32(2),
           ymin   = cms.double(0.5),
           ymax   = cms.double(2.5)
    ),
    DigiChargeH = cms.PSet(
      Nbins = cms.int32(261),
      xmin   = cms.double(0.5),
      xmax   = cms.double(260.5)
    ), 
    NumberOfClustersH = cms.PSet(
           Nbins = cms.int32(51),
           xmin = cms.double(-0.5),
           xmax = cms.double(50.5)
    ),
    ClusterWidthH = cms.PSet(
           Nbins = cms.int32(16),
           xmin   = cms.double(-0.5),
           xmax   = cms.double(15.5),
    ),
    ClusterChargeH = cms.PSet(
      Nbins = cms.int32(1024),
      xmin   = cms.double(0.5),
      xmax   = cms.double(1024.5)
    ),  
    ClusterPositionH = cms.PSet(
      Nbins = cms.int32(260),
      xmin   = cms.double(0.5),
      xmax   = cms.double(260.5)
    )  
)
