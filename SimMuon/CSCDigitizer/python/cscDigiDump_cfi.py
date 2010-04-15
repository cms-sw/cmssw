import FWCore.ParameterSet.Config as cms

cscDigiDump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
)



