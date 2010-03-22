import FWCore.ParameterSet.Config as cms
  
process = cms.Process("TOPVAL")

## define MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Core.DQMStore_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

## source Input File
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
   ## the following files are in for testing
   '/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/14920B0A-0DE8-DE11-B138-002618943926.root'
  )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))

## define output options
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

