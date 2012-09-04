import FWCore.ParameterSet.Config as cms

globalhitsprodhist = cms.EDProducer("GlobalHitsProdHist",
    G4VtxSrc = cms.InputTag("g4SimHits"),
    G4TrkSrc = cms.InputTag("g4SimHits"),
    MuonRpcSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
    SiTOBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),
    SiTECHighSrc = cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalEESrc = cms.InputTag("g4SimHits","EcalHitsEE"),
    SiTIBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
    SiTECLowSrc = cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),
    MuonCscSrc = cms.InputTag("g4SimHits","MuonCSCHits"),
    SiTIDHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),
    Name = cms.untracked.string('GlobalHitsProdHist'),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    SiTIBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),
    SiTOBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    ECalESSrc = cms.InputTag("g4SimHits","EcalHitsES"),
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
    MuonDtSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("g4SimHits","EcalHitsEB")
)


