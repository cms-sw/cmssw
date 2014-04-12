import FWCore.ParameterSet.Config as cms

globalhits = cms.EDProducer("GlobalHitsProducer",
    G4VtxSrc = cms.InputTag("g4SimHits"),
    G4TrkSrc = cms.InputTag("g4SimHits"),
    MuonRpcSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Label = cms.string('GlobalHits'),
    SiTOBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),
    SiTECHighSrc = cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalEESrc = cms.InputTag("g4SimHits","EcalHitsEE"),
    SiTIBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
    SiTECLowSrc = cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),
    MuonCscSrc = cms.InputTag("g4SimHits","MuonCSCHits"),
    SiTIDHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),
    Name = cms.untracked.string('GlobalHitsProducer'),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
    # as of 090p3 should be g4SimHits. Anything earlier SimG4Object
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    SiTIBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),
    SiTOBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
    ECalESSrc = cms.InputTag("g4SimHits","EcalHitsES"),
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
    MuonDtSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    Frequency = cms.untracked.int32(50),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    ECalEBSrc = cms.InputTag("g4SimHits","EcalHitsEB")
)


