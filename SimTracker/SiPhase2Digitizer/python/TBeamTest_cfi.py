import FWCore.ParameterSet.Config as cms

tbeamTest = cms.EDAnalyzer("TBeamTest",
    TopFolderName = cms.string("TBeamTest"),
    OuterTrackerDigiSource = cms.InputTag("mix", "Tracker"),
    OuterTrackerDigiSimSource = cms.InputTag("simSiPixelDigis", "Tracker"),
    SimTrackSource = cms.InputTag("g4SimHits"),
    GeometryType = cms.string('idealForDigi'),
    PhiAngles= cms.vdouble(0, 1.3, 2, 3.1, 4.2, 5.0, 6.2, 7.0, 8.1, 8.5, 9.2, 9.9, 10.5, 11.0, 11.5, 12.5, 13.3, 13.9, 15),
    NumberOfDigisH = cms.PSet(
           Nbins = cms.int32(200),
           xmin = cms.double(-0.5),
           xmax = cms.double(200.5)
    ),
    PositionOfDigisH = cms.PSet(
           Nxbins = cms.int32(1016),
           xmin   = cms.double(0.5),
           xmax   = cms.double(1016.5),
           Nybins = cms.int32(10),
           ymin   = cms.double(0.5),
           ymax   = cms.double(10.5)
    ),
    DigiChargeH = cms.PSet(
      Nbins = cms.int32(261),
      xmin   = cms.double(0.5),
      xmax   = cms.double(260.5)
    ), 
    NumberOfClustersH = cms.PSet(
           Nbins = cms.int32(200),
           xmin = cms.double(-0.5),
           xmax = cms.double(200.5)
    ),
    ClusterWidthH = cms.PSet(
           Nbins = cms.int32(16),
           xmin   = cms.double(-0.5),
           xmax   = cms.double(15.5),
    ),
    ClusterPositionH = cms.PSet(
        Nbins = cms.int32(1016),
        xmin   = cms.double(0.5),
        xmax   = cms.double(1016.5)
    )
)
