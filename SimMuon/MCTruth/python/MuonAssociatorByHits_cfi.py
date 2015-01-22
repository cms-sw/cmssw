import FWCore.ParameterSet.Config as cms


muonAssociatorByHitsCommonParameters = cms.PSet(
    dumpInputCollections = cms.untracked.bool(False),
    #
    #....... general input parameters
    #
    # include invalid muon hits
    includeZeroHitMuons = cms.bool(True),
    #
    # accept to match only tracker/muon stub of globalMuons
    acceptOneStubMatchings = cms.bool(True),
    #
    # switches to be set according to the input Track collection
    UseTracker = cms.bool(True),
    UseMuon = cms.bool(True),
    #
    # cuts for the muon stub
    AbsoluteNumberOfHits_muon = cms.bool(False),
    NHitCut_muon = cms.uint32(0),
    EfficiencyCut_muon = cms.double(0.),
    PurityCut_muon = cms.double(0.),
    #
    # cuts for the tracker stub
    AbsoluteNumberOfHits_track = cms.bool(False),
    NHitCut_track = cms.uint32(0),
    EfficiencyCut_track = cms.double(0.),
    PurityCut_track = cms.double(0.),
    #
    # switches for the tracker stub
    UsePixels = cms.bool(True),
    UseGrouped = cms.bool(True),
    UseSplitting = cms.bool(True),
    ThreeHitTracksAreSpecial = cms.bool(False),
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
        'TrackerHitsPixelEndcapHighTof'),
    #
    # to associate to reco::Muon segments (3.5.X only)
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection = cms.InputTag("cscSegments"),
)

muonAssociatorByHits = cms.EDProducer("MuonAssociatorEDProducer",
    # COMMON CONFIGURATION
    muonAssociatorByHitsCommonParameters,
    # for Muon Track association
    #
    #     input collections
    #
    # ... reco::Track collection
    # tracksTag = cms.InputTag("standAloneMuons"),
    # tracksTag = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    # tracksTag = cms.InputTag("standAloneSETMuons"),
    # tracksTag = cms.InputTag("standAloneSETMuons","UpdatedAtVtx"),                                   
    # tracksTag = cms.InputTag("cosmicMuons"),
    tracksTag = cms.InputTag("globalMuons"),
    # tracksTag = cms.InputTag("tevMuons","firstHit"),
    # tracksTag = cms.InputTag("tevMuons","picky"),                                     
    # tracksTag = cms.InputTag("globalSETMuons"),
    # tracksTag = cms.InputTag("globalCosmicMuons"),
    # tracksTag = cms.InputTag("generalTracks"),
    # tracksTag = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    # tracksTag = cms.InputTag("hltL2Muons"),
    # tracksTag = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    # tracksTag = cms.InputTag("hltL3Muons")
    # tracksTag = cms.InputTag("hltL3Muons","L2Seeded")
    # tracksTag = cms.InputTag("hltL3TkTracksFromL2")
    #
    # ... TrackingParticle collection
    tpTag = cms.InputTag("mix","MergedTrackTruth"),
    ignoreMissingTrackCollection = cms.untracked.bool(False),
)
 
