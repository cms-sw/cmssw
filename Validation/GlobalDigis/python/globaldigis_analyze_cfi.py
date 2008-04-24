import FWCore.ParameterSet.Config as cms

globaldigisanalyze = cms.EDAnalyzer("GlobalDigisAnalyzer",
    MuCSCStripSrc = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("simMuonDTDigis"),
    Name = cms.untracked.string('GlobalDigisAnalyzer'),
    MuCSCWireSrc = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    ECalEESrc = cms.InputTag("simEcalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    SiPxlSrc = cms.InputTag("simSiPixelDigis"),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Frequency = cms.untracked.int32(50),
    MuRPCSrc = cms.InputTag("simMuonRPCDigis"),
    ECalEBSrc = cms.InputTag("simEcalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("simEcalPreshowerDigis"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    #InputTag HCalDigi  = hcalUnsuppressedDigis
    HCalDigi = cms.InputTag("simHcalDigis")
)


