import FWCore.ParameterSet.Config as cms

globaldigisanalyze = cms.EDAnalyzer("GlobalDigisAnalyzer",
    MuCSCStripSrc = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("muonDTDigis"),
    Name = cms.untracked.string('GlobalDigisAnalyzer'),
    MuCSCWireSrc = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    ECalEESrc = cms.InputTag("ecalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("siStripDigis","ZeroSuppressed"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    SiPxlSrc = cms.InputTag("siPixelDigis"),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Frequency = cms.untracked.int32(50),
    MuRPCSrc = cms.InputTag("muonRPCDigis"),
    ECalEBSrc = cms.InputTag("ecalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("ecalPreshowerDigis"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    #InputTag HCalDigi  = hcalUnsuppressedDigis
    HCalDigi = cms.InputTag("hcalDigis")
)


