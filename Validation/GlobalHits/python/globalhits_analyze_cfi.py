import FWCore.ParameterSet.Config as cms

globalhitsanalyze = cms.EDAnalyzer("GlobalHitsAnalyzer",
    MuonRpcSrc = cms.InputTag("g4SimHits","MuonRPCHits",""),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof",""),
    SiTOBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTOBLowTof",""),
    SiTECHighSrc = cms.InputTag("g4SimHits","TrackerHitsTECHighTof",""),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof",""),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits",""),
    ECalEESrc = cms.InputTag("g4SimHits","EcalHitsEE",""),
    SiTIBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIBHighTof",""),
    SiTECLowSrc = cms.InputTag("g4SimHits","TrackerHitsTECLowTof",""),
    MuonCscSrc = cms.InputTag("g4SimHits","MuonCSCHits",""),
    SiTIDHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIDHighTof",""),
    Name = cms.untracked.string('GlobalHitsAnalyzer'),
    Verbosity = cms.untracked.int32(0),
    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof",""),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof",""),
    SiTIBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIBLowTof",""),
    SiTOBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTOBHighTof",""),
    VtxUnit = cms.untracked.int32(1),
    ECalESSrc = cms.InputTag("g4SimHits","EcalHitsES",""),
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof",""),
    OutputFile = cms.string('GlobalHitsHistogramsAnalyze.root'),
    MuonDtSrc = cms.InputTag("g4SimHits","MuonDTHits",""),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("g4SimHits","EcalHitsEB","")
)




