import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
digiMon = DQMEDAnalyzer('Phase2TrackerMonitorDigi',
    Verbosity = cms.bool(False),
    TopFolderName = cms.string("Ph2TkDigi"),
    PixelPlotFillingFlag = cms.bool(False),
    InnerPixelDigiSource   = cms.InputTag("simSiPixelDigis","Pixel"),
    OuterTrackerDigiSource = cms.InputTag("mix", "Tracker"),
    GeometryType = cms.string('idealForDigi'),
    NumberOfDigisPerDetH = cms.PSet(
           Nbins = cms.int32(100),
           xmin = cms.double(-0.5),
           xmax = cms.double(99.5)
    ),
    DigiOccupancySH = cms.PSet(
           Nbins = cms.int32(51),
           xmin = cms.double(-0.001),
           xmax = cms.double(0.05)
    ),
    DigiOccupancyPH = cms.PSet(
           Nbins = cms.int32(51),
           xmin = cms.double(-0.0001),
           xmax = cms.double(0.005)
    ),
    PositionOfDigisH = cms.PSet(
           Nxbins = cms.int32(1016),
           xmin   = cms.double(0.5),
           xmax   = cms.double(1016.5),
           Nybins = cms.int32(10),
           ymin   = cms.double(0.5),
           ymax   = cms.double(10.5)
    ),
    EtaH = cms.PSet(
        Nbins  = cms.int32(45),
        xmin   = cms.double(-4.5),
        xmax   = cms.double(4.5)
    ),
    DigiChargeH = cms.PSet(
      Nbins = cms.int32(261),
      xmin   = cms.double(0.5),
      xmax   = cms.double(260.5)
    ), 
    TotalNumberOfDigisPerLayerH = cms.PSet(
      Nbins = cms.int32(100),
      xmin   = cms.double(0.0),
      xmax   = cms.double(50000.0)
    ),
    NumberOfHitDetsPerLayerH = cms.PSet(
      Nbins = cms.int32(2000),
      xmin   = cms.double(-0.5),
      xmax   = cms.double(2000.5)
    ),
    NumberOfClustersPerDetH = cms.PSet(
           Nbins = cms.int32(100),
           xmin = cms.double(-0.5),
           xmax = cms.double(99.5)
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
        Nbins = cms.int32(1016),
        xmin   = cms.double(0.5),
        xmax   = cms.double(1016.5)
    ),  
    XYPositionMapH = cms.PSet(
        Nxbins = cms.int32(1250),
        xmin   = cms.double(-1250.),
        xmax   = cms.double(1250.),
        Nybins = cms.int32(1250),
        ymin   = cms.double(-1250.),
        ymax   = cms.double(1250.)
    ),
    RZPositionMapH = cms.PSet(
        Nxbins = cms.int32(3000),
        xmin   = cms.double(-3000.),
        xmax   = cms.double(3000.),
        Nybins = cms.int32(1250),
        ymin   = cms.double(0.),
        ymax   = cms.double(1250.)
    )
)
