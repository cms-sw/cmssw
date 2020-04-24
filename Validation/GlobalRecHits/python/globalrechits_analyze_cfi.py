import FWCore.ParameterSet.Config as cms

globalrechitsanalyze = cms.EDAnalyzer("GlobalRecHitsAnalyzer",
    MuDTSrc = cms.InputTag("dt1DRecHits"),
    SiPxlSrc = cms.InputTag("siPixelRecHits"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    associateRecoTracks = cms.bool(False),
    MuDTSimSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    # needed for TrackerHitAssociator
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof', 
        'g4SimHitsTrackerHitsTIBHighTof', 
        'g4SimHitsTrackerHitsTIDLowTof', 
        'g4SimHitsTrackerHitsTIDHighTof', 
        'g4SimHitsTrackerHitsTOBLowTof', 
        'g4SimHitsTrackerHitsTOBHighTof', 
        'g4SimHitsTrackerHitsTECLowTof', 
        'g4SimHitsTrackerHitsTECHighTof', 
        'g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof'),
    ECalEESrc = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    MuRPCSimSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    SiStripSrc = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalESSrc = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    hitsProducer = cms.string('g4SimHits'),
    ECalUncalEESrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    Name = cms.untracked.string('GlobalRecHitsAnalyzer'),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),

    associateStrip = cms.bool(True),
    MuRPCSrc = cms.InputTag("rpcRecHits"),
    ECalUncalEBSrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    MuCSCSrc = cms.InputTag("csc2DRecHits"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)



