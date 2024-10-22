import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
globaldigisanalyze = DQMEDAnalyzer('GlobalDigisAnalyzer',
    hitsProducer = cms.string('g4SimHits'),
    MuCSCStripSrc = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("simMuonDTDigis"),
    Name = cms.untracked.string('GlobalDigisAnalyzer'),
    SiPxlSrc = cms.InputTag("simSiPixelDigis"),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    MuCSCWireSrc = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    ECalEESrc = cms.InputTag("simEcalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    Frequency = cms.untracked.int32(50),
    MuRPCSrc = cms.InputTag("simMuonRPCDigis"),
    ECalEBSrc = cms.InputTag("simEcalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("simEcalPreshowerDigis"),
    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    #InputTag HCalDigi  = simHcalUnsuppressedDigis
    HCalDigi = cms.InputTag("simHcalDigis")
)
