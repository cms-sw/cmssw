import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDigiAnalyzerFromDigi")

# geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_31X::All"

# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")
#process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/0C6D25CB-524F-DE11-B3AA-0030487A1FEC.root')
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/F20698E4-834E-DE11-AD06-001D09F2441B.root')
)

# Now this piece of code is in Validation/MuonDTDigis/python/dtDigiValidation_cfi.py
# process.muondtdigianalyzer = cms.EDFilter("MuonDTDigis",
#    DigiLabel = cms.untracked.string('muonDTDigis'),
#    SimHitLabel = cms.untracked.string('g4SimHits'),
#    outputFile = cms.untracked.string('DTDigiPlots.root'),
#    verbose = cms.untracked.bool(True)
# )

# DT Muon Digis validation sequence
process.load("Validation.MuonDTDigis.dtDigiValidation_cfi")

process.p = cms.Path(process.muondtdigianalyzer)


