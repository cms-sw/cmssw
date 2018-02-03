import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
digiValid = DQMEDAnalyzer('Phase2TrackerValidateDigi',
    Verbosity = cms.bool(False),
    TopFolderName = cms.string("Ph2TkPixelDigi"),
    PixelPlotFillingFlag = cms.bool(False),
    OuterTrackerDigiSource = cms.InputTag("mix", "Tracker"),
    OuterTrackerDigiSimLinkSource = cms.InputTag("simSiPixelDigis", "Tracker"),
    InnerPixelDigiSource   = cms.InputTag("simSiPixelDigis","Pixel"),                          
    InnerPixelDigiSimLinkSource = cms.InputTag("simSiPixelDigis", "Pixel"), 
    PSimHitSource  = cms.VInputTag('g4SimHits:TrackerHitsPixelBarrelLowTof',
                                   'g4SimHits:TrackerHitsPixelBarrelHighTof',
                                   'g4SimHits:TrackerHitsPixelEndcapLowTof',
                                   'g4SimHits:TrackerHitsPixelEndcapHighTof',
                                   'g4SimHits:TrackerHitsTIBLowTof',
                                   'g4SimHits:TrackerHitsTIBHighTof',
                                   'g4SimHits:TrackerHitsTIDLowTof',
                                   'g4SimHits:TrackerHitsTIDHighTof',
                                   'g4SimHits:TrackerHitsTOBLowTof',
                                   'g4SimHits:TrackerHitsTOBHighTof',
                                   'g4SimHits:TrackerHitsTECLowTof',
                                   'g4SimHits:TrackerHitsTECHighTof'),
    SimTrackSource = cms.InputTag("g4SimHits"),
    SimVertexSource = cms.InputTag("g4SimHits"),
    GeometryType = cms.string('idealForDigi'),
    PtCutOff      = cms.double(9.5),                           
    EtaCutOff      = cms.double(3.5),                           
    TOFLowerCutOff = cms.double(-12.5),
    TOFUpperCutOff = cms.double(12.5),
    TrackPtH = cms.PSet(
        Nbins  = cms.int32(50),
        xmin   = cms.double(0.0),
        xmax   = cms.double(100.0)
    ),
    TrackEtaH = cms.PSet(
        Nbins  = cms.int32(45),
        xmin   = cms.double(-4.5),
        xmax   = cms.double(4.5)
    ),
    TrackPhiH = cms.PSet(
        Nbins  = cms.int32(64),
        xmin   = cms.double(-3.2),
        xmax   = cms.double(3.2)
    ),  
    SimHitElossH = cms.PSet(
        Nbins  = cms.int32(100),
        xmin   = cms.double(0.0),
        xmax   = cms.double(100000.0)
    ),  
    SimHitDxH = cms.PSet(
        Nbins  = cms.int32(1000),
        xmin   = cms.double(0.0),
        xmax   = cms.double(0.1)
    ),
    SimHitDyH = cms.PSet(
        Nbins  = cms.int32(1000),
        xmin   = cms.double(0.0),
        xmax   = cms.double(0.1)
    ),
    SimHitDzH = cms.PSet(
        Nbins  = cms.int32(150),
        xmin   = cms.double(0.0),
        xmax   = cms.double(0.03)
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
    ),
   TOFEtaMapH = cms.PSet(
        Nxbins = cms.int32(45),
        xmin   = cms.double(-4.5),
        xmax   = cms.double(4.5),
        Nybins = cms.int32(100),
        ymin   = cms.double(0.),
        ymax   = cms.double(50.)
    ),
   TOFPhiMapH = cms.PSet(
        Nxbins = cms.int32(64),
        xmin   = cms.double(-3.2),
        xmax   = cms.double(3.2),
        Nybins = cms.int32(100),
        ymin   = cms.double(0.),
        ymax   = cms.double(50.)
    ),
   TOFZMapH = cms.PSet(
        Nxbins = cms.int32(3000),
        xmin   = cms.double(-300.),
        xmax   = cms.double(300.),
        Nybins = cms.int32(100),
        ymin   = cms.double(0.),
        ymax   = cms.double(50.)
    ),
    TOFRMapH = cms.PSet(
        Nxbins = cms.int32(1200),
        xmin   = cms.double(0.),
        xmax   = cms.double(120.),
        Nybins = cms.int32(100),
        ymin   = cms.double(0.),
        ymax   = cms.double(50.)
    )
)
