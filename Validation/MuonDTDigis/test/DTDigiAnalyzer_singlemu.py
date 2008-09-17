import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDigiAnalyzerFromDigi")

# geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")
process.load("Configuration.StandardSequences.FakeConditions_cff")

# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")
#process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
 #  fileNames = cms.untracked.vstring('/store/mc/2006/12/21/mc-physval-120-SingleMuPlus-Pt100/0000/04B074BD-B496-DB11-B4BF-00096BB5BFD2.root')
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/400B03E1-2A6C-DD11-A78C-001617E30D4A.root')
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


