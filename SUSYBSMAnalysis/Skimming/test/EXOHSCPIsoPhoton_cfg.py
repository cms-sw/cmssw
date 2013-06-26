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
process.GlobalTag.globaltag = 'GR_R_38X_V14::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root',
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/EEAA24FA-25B4-DF11-A5F1-000423D98950.root',
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/C40EDB4E-1DB4-DF11-A83C-0030487C90C2.root',
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/C2497931-2CB4-DF11-A92C-003048F1183E.root',
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/AC68ABE0-19B4-DF11-BB93-0030487C7E18.root',
       '/store/data/Run2010A/EG/RECO/v4/000/144/114/92F10BD6-22B4-DF11-B5FC-0030487CD812.root',


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


#process.exoticaHSCPSkimPath=cms.Path(process.exoticaHSCPSeq)
process.exoticaHSCPSkimPath=cms.Path(process.exoticaHSCPIsoPhotonSeq)
process.exoticaHSCPOutputModule.fileName = cms.untracked.string('EXOHSCP.root')

process.endPath = cms.EndPath(process.exoticaHSCPOutputModule)





