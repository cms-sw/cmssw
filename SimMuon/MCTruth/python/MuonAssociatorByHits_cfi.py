import FWCore.ParameterSet.Config as cms

muonAssociatorByHits = cms.EDProducer("MuonAssociatorEDProducer",
    RPCdigisimlinkTag = cms.InputTag("simMuonRPCDigis","RPCDigiSimLink"),
    EfficiencyCut_track = cms.double(0.5),
    UseSplitting = cms.bool(True),
    DTdigiTag = cms.InputTag("simMuonDTDigis"),
    associatorByWire = cms.bool(False),
    EfficiencyCut_muon = cms.double(0.5),
    ThreeHitTracksAreSpecial = cms.bool(True),
    #    InputTag tracksTag = standAloneMuons:UpdatedAtVtx
    #    InputTag tracksTag = globalMuons
    # ... TrackingParticle collection 
    tpTag = cms.InputTag("mergedtruth","MergedTrackTruth"),
    dumpDT = cms.bool(False),
    # for DT Hit associator
    crossingframe = cms.bool(True),
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
    UseGrouped = cms.bool(True),
    #....... general input parameters
    #
    AbsoluteNumberOfHits_track = cms.bool(False),
    dumpInputCollections = cms.bool(False),
    DTdigisimlinkTag = cms.InputTag("simMuonDTDigis"),
    PurityCut_track = cms.double(0.5),
    MinHitCut_muon = cms.uint32(1),
    CSCwireLinksTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigiSimLinks"),
    # for CSC Hit associator
    CSCsimHitsXFTag = cms.InputTag("mix","g4SimHitsMuonCSCHits"),
    DTsimhitsTag = cms.InputTag("g4SimHits","MuonDTHits"),
    MinHitCut_track = cms.uint32(1),
    simtracksTag = cms.InputTag("g4SimHits"),
    CSClinksTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigiSimLinks"),
    # for Tracker Hit associator
    #
    associatePixel = cms.bool(True),
    SimToReco_useTracker = cms.bool(False),
    simtracksXFTag = cms.InputTag("mix","g4SimHits"),
    associateStrip = cms.bool(True),
    DTrechitTag = cms.InputTag("dt1DRecHits"),
    # for RPC Hit associator
    RPCsimhitsXFTag = cms.InputTag("mix","g4SimHitsMuonRPCHits"),
    SimToReco_useMuon = cms.bool(True),
    associateRecoTracks = cms.bool(True),
    # for Muon Track association
    #   input collections
    # ... reco::Track collection 
    tracksTag = cms.InputTag("standAloneMuons"),
    #........(for inner tracker stub of Global Muons)...
    UsePixels = cms.bool(True),
    AbsoluteNumberOfHits_muon = cms.bool(False),
    links_exist = cms.bool(True),
    DTsimhitsXFTag = cms.InputTag("mix","g4SimHitsMuonDTHits"),
    PurityCut_muon = cms.double(0.5)
)



