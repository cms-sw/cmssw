import FWCore.ParameterSet.Config as cms

cscDigiValidation = cms.EDFilter("CSCDigiValidation",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    outputFile = cms.string('CSCDigiValidation.root'),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    doSim = cms.bool(False)
)



