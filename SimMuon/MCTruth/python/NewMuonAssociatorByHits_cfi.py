import FWCore.ParameterSet.Config as cms


NewMuonAssociatorByHitsCommonParameters = cms.PSet(
    dumpInputCollections = cms.untracked.bool(False),
    #
    #....... general input parameters
    #
    # include invalid muon hits
    includeZeroHitMuons = cms.bool(True),
    #
    # accept to match only tracker/muon stub of globalMuons
    acceptOneStubMatchings = cms.bool(False),
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
    # for GEM Hit associator
    useGEMs = cms.bool(False),
    GEMsimhitsTag = cms.InputTag("g4SimHits","MuonGEMHits"),
    GEMsimhitsXFTag = cms.InputTag("mix","g4SimHitsMuonGEMHits"),
    GEMdigisimlinkTag = cms.InputTag("simMuonGEMDigis","GEM"),
    #
    # for Tracker Hit associator
    #
    associatePixel = cms.bool(True),
    associateStrip = cms.bool(True),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
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
    # to associate to reco::Muon segments 
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection = cms.InputTag("cscSegments"),
)


from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
#if True:
    obj = NewMuonAssociatorByHitsCommonParameters
    obj.simtracksTag = "famosSimHits"
    obj.DTsimhitsTag  = "MuonSimHits:MuonDTHits"
    obj.CSCsimHitsTag = "MuonSimHits:MuonCSCHits"
    obj.RPCsimhitsTag = "MuonSimHits:MuonRPCHits"
    obj.simtracksXFTag = "mix:famosSimHits"
    obj.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
    obj.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
    obj.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
    obj.ROUList = ['famosSimHitsTrackerHits']

  
NewMuonAssociatorByHits = cms.EDProducer("MuonAssociatorEDProducer",
    # COMMON CONFIGURATION
    NewMuonAssociatorByHitsCommonParameters,
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

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( NewMuonAssociatorByHits, useGEMs = cms.bool(True) )
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( NewMuonAssociatorByHits, pixelSimLinkSrc = "simSiPixelDigis:Pixel" )
phase2_tracker.toModify( NewMuonAssociatorByHits, stripSimLinkSrc = "simSiPixelDigis:Tracker" )

