import FWCore.ParameterSet.Config as cms

muonAssociatorByHits = cms.EDProducer("MuonAssociatorEDProducer",
    # for Muon Track association
    #
    #     input collections
    #
    # ... reco::Track collection
    tracksTag = cms.InputTag("standAloneMuons"),
    # tracksTag = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    # tracksTag = cms.InputTag("globalMuons"),
    # tracksTag = cms.InputTag("generalTracks"),
    # tracksTag = cms.InputTag("hltL2Muons"),
    # tracksTag = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    # tracksTag = cms.InputTag("hltL3Muons")
    #
    # ... TrackingParticle collection
    tpTag = cms.InputTag("mergedtruth","MergedTrackTruth"),
    #
    ignoreMissingTrackCollection = cms.untracked.bool(False),
    dumpInputCollections = cms.bool(False),
    #
    #....... general input parameters
    #
    AbsoluteNumberOfHits_track = cms.bool(False),
    MinHitCut_track = cms.uint32(1),
    AbsoluteNumberOfHits_muon = cms.bool(False),
    MinHitCut_muon = cms.uint32(1),
    #
    UseTracker = cms.bool(True),
    UseMuon = cms.bool(True),
    #
    PurityCut_track = cms.double(0.5),
    PurityCut_muon = cms.double(0.5),
    #
    EfficiencyCut_track = cms.double(0.5),
    EfficiencyCut_muon = cms.double(0.5),
    #
    #........(for inner tracker stub of Global Muons)...
    UsePixels = cms.bool(True),
    UseGrouped = cms.bool(True),
    UseSplitting = cms.bool(True),
    ThreeHitTracksAreSpecial = cms.bool(True),
    #
    # for DT Hit associator
    crossingframe = cms.bool(False),
    simtracksTag = cms.InputTag("g4SimHits"),
    simtracksXFTag = cms.InputTag("mix","g4SimHits"),
    #
    DTsimhitsTag = cms.InputTag("g4SimHits","MuonDTHits"),
    DTsimhitsXFTag = cms.InputTag("mix","g4SimHitsMuonDTHits"),
    DTdigiTag = cms.InputTag("simMuonDTDigis"),
    DTdigisimlinkTag = cms.InputTag("simMuonDTDigis"),
    DTrechitTag = cms.InputTag("dt1DRecHits"),
    #
    dumpDT = cms.bool(False),
    links_exist = cms.bool(True),
    associatorByWire = cms.bool(False),
    #
    # for CSC Hit associator
    CSCsimHitsTag = cms.InputTag("g4SimHits","MuonCSCHits"),
    CSCsimHitsXFTag = cms.InputTag("mix","g4SimHitsMuonCSCHits"),
    CSClinksTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigiSimLinks"),
    CSCwireLinksTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigiSimLinks"),
    #
    # for RPC Hit associator
    RPCsimhitsTag = cms.InputTag("g4SimHits","MuonRPCHits"),
    RPCsimhitsXFTag = cms.InputTag("mix","g4SimHitsMuonRPCHits"),
    RPCdigisimlinkTag = cms.InputTag("simMuonRPCDigis","RPCDigiSimLink"),
    #
    # for Tracker Hit associator
    #
    associatePixel = cms.bool(True),
    associateStrip = cms.bool(True),
    associateRecoTracks = cms.bool(True),
    #                                
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
        'TrackerHitsPixelEndcapHighTof')
)



