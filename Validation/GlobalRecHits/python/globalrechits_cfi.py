import FWCore.ParameterSet.Config as cms

globalrechits = cms.EDProducer("GlobalRecHitsProducer",
    MuDTSrc = cms.InputTag("dt1DRecHits"),
    SiPxlSrc = cms.InputTag("siPixelRecHits"),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Label = cms.string('GlobalRecHits'),
    associateRecoTracks = cms.bool(False),
    MuDTSimSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    # needed for TrackerHitAssociator
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof', 
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    MuRPCSrc = cms.InputTag("rpcRecHits"),
    ECalEESrc = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    MuRPCSimSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    SiStripSrc = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalESSrc = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    ECalUncalEESrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    Name = cms.untracked.string('GlobalRecHitsProducer'),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    associateStrip = cms.bool(True),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    ECalUncalEBSrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    MuCSCSrc = cms.InputTag("csc2DRecHits"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


