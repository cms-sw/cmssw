import FWCore.ParameterSet.Config as cms

globaldigisanalyze = cms.EDAnalyzer("GlobalDigisAnalyzer",
    MuCSCStripSrc = cms.InputTag("muonCSCDigis","MuonCSCStripDigi",""),
    outputFile = cms.string('GlobalDigisHistogramsAnalyze.root'),
    DoOutput = cms.bool(False),
    Name = cms.untracked.string('GlobalDigisAnalyzer'),
    MuCSCWireSrc = cms.InputTag("muonCSCDigis","MuonCSCWireDigi",""),
    Verbosity = cms.untracked.int32(0),
    ECalEESrc = cms.InputTag("ecalDigis","eeDigis",""),
    SiStripSrc = cms.InputTag("siStripDigis","",""),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits",""),
    SiPxlSrc = cms.InputTag("siPixelDigis","",""),
    MuDTSrc = cms.InputTag("muonDTDigis","",""),
    VtxUnit = cms.untracked.int32(1),
    ECalEBSrc = cms.InputTag("ecalDigis","ebDigis",""),
    ECalESSrc = cms.InputTag("ecalPreshowerDigis","",""),
    Frequency = cms.untracked.int32(50),
    HCalDigi = cms.InputTag("hcalUnsuppressedDigis","",""),
    MuRPCSrc = cms.InputTag("muonRPCDigis","","")
)




