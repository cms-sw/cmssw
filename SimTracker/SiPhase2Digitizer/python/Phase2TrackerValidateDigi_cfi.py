import FWCore.ParameterSet.Config as cms

digiValid = cms.EDAnalyzer("Phase2TrackerValidateDigi",
    Verbosity = cms.bool(False),
    TopFolderName = cms.string("Phase2Tracker"),
    PixelDigiSource    = cms.InputTag("simSiPixelDigis","Pixel"),                          
    OuterTrackerDigiSource    = cms.InputTag("mix", "Tracker"),                          
    DigiSimLinkSource    = cms.InputTag("simSiPixelDigis"),                          
    PtCutOff      = cms.double(10.0),                           
    EtaCutOff      = cms.double(2.5),                           
    TrackPtH = cms.PSet(
        Nbins  = cms.int32(50),
        xmin   = cms.double(0.0),
        xmax   = cms.double(100.0)
    ),
    TrackEtaH = cms.PSet(
        Nbins  = cms.int32(25),
        xmin   = cms.double(-2.5),
        xmax   = cms.double(2.5),
    ),
    TrackPhiH = cms.PSet(
        Nbins  = cms.int32(64),
        xmin   = cms.double(-3.2),
        xmax   = cms.double(3.2)
    ) 
)
