import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOHSCPSkim")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START3X_V26::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
   )
)

#load the EventContent and Skim cff/i files for EXOHSCP sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')

#possible event content modification.
#by uncomment below.
## from Configuration.EventContent.EventContent_cff import *
## from SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi import *
## SpecifiedEvenetContent=cms.PSet(
##     outputCommands = cms.untracked.vstring(
##       "drop *",
##       "keep L1GlobalTriggerReadoutRecord_*_*_*",
##       "keep recoVertexs_offlinePrimaryVertices_*_*",
##       "keep recoMuons_muons_*_*",
## 	  )
## 	)
## process.exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)

#possible change skim cut.
## process.generalTracksSkim.ptMin = cms.double(10)
## process.reducedHSCPhbhereco.TrackPt=cms.double(15)


process.exoticaHSCPSkimPath=cms.Path(process.exoticaHSCPSeq)
process.exoticaHSCPOutputModule.fileName = cms.untracked.string('EXOHSCP.root')

process.endPath = cms.EndPath(process.exoticaHSCPOutputModule)





