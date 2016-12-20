import FWCore.ParameterSet.Config as cms

digiValid = cms.EDAnalyzer("Phase2TrackerValidateDigi",
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
    PtCutOff      = cms.double(10.0),                           
    EtaCutOff      = cms.double(3.5),                           
    TrackPtH = cms.PSet(
        Nbins  = cms.int32(50),
        xmin   = cms.double(0.0),
        xmax   = cms.double(100.0)
    ),
    TrackEtaH = cms.PSet(
        Nbins  = cms.int32(35),
        xmin   = cms.double(-3.5),
        xmax   = cms.double(3.5),
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
    XYPositionMapH = cms.PSet(
           Nxbins = cms.int32(1200),
           xmin   = cms.double(-120.),
           xmax   = cms.double(120.),
           Nybins = cms.int32(1200),
           ymin   = cms.double(-120.),
           ymax   = cms.double(120.)
    ),
    RZPositionMapH = cms.PSet(
           Nxbins = cms.int32(3000),
           xmin   = cms.double(-300.),
           xmax   = cms.double(300.),
           Nybins = cms.int32(1200),
           ymin   = cms.double(0.),
           ymax   = cms.double(120.)
    )
)
