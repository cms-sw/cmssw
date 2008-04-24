import FWCore.ParameterSet.Config as cms

globaldigis = cms.EDProducer("GlobalDigisProducer",
    MuCSCStripSrc = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("simMuonDTDigis"),
    Name = cms.untracked.string('GlobalDigisProducer'),
    MuCSCWireSrc = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    ECalEESrc = cms.InputTag("simEcalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("simSiStripDigis"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    SiPxlSrc = cms.InputTag("simSiPixelDigis"),
    Frequency = cms.untracked.int32(50),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Label = cms.string('GlobalDigis'),
    ECalEBSrc = cms.InputTag("simEcalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("simEcalPreshowerDigis"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    HCalDigi = cms.InputTag("simHcalUnsuppressedDigis")
)


