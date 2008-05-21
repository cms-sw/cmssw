import FWCore.ParameterSet.Config as cms

process = cms.Process("SimDigiDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myfile.root')
)

process.prod = cms.EDFilter("SimDigiDumper",
    MuCSCStripSrc = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("muonDTDigis"),
    HCalDigi = cms.InputTag("hcalDigis"),
    MuCSCWireSrc = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    ECalEESrc = cms.InputTag("ecalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("siStripDigis","ZeroSuppressed"),
    SiPxlSrc = cms.InputTag("siPixelDigis"),
    ECalEBSrc = cms.InputTag("ecalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("ecalPreshowerDigis"),
    MuRPCSrc = cms.InputTag("muonRPCDigis")
)

process.p1 = cms.Path(process.prod)


