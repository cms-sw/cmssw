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

process.prod = cms.EDAnalyzer("SimDigiDumper",
    MuCSCStripSrc = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("simMuonDTDigis"),
    HCalDigi = cms.InputTag("simHcalDigis"),
    ZdcDigi = cms.InputTag("simHcalUnsuppressedDigis"),                        
    MuCSCWireSrc = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    ECalEESrc = cms.InputTag("simEcalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    SiPxlSrc = cms.InputTag("simSiPixelDigis"),
    ECalEBSrc = cms.InputTag("simEcalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("simEcalPreshowerDigis"),
    MuRPCSrc = cms.InputTag("simMuonRPCDigis")
)

process.p1 = cms.Path(process.prod)


