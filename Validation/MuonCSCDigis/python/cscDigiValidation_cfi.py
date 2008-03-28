import FWCore.ParameterSet.Config as cms

cscDigiValidation = cms.EDFilter("CSCDigiValidation",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    outputFile = cms.string('CSCDigiValidation.root'),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    doSim = cms.bool(False)
)


