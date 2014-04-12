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
'/store/data/Run2010A/Mu/RECO/v4/000/144/114/9C954151-32B4-DF11-BB88-001D09F27003.root'
# '/store/mc/Spring10/Wmunu_Wplus-powheg/GEN-SIM-RECO/START3X_V26_S09-v1/0024/FAC26AD5-5148-DF11-BC3D-90E6BA0D09AC.root',


   )
)

#load the EventContent and Skim cff/i files for EXOHSCP sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')

#possible event content modification.
#by uncomment below.
from Configuration.EventContent.EventContent_cff import *
from SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi import *
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep recoWMuNuCandidates_*_*_*",
	  )
	)
process.exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)

#possible change skim cut.
## process.generalTracksSkim.ptMin = cms.double(10)
## process.reducedHSCPhbhereco.TrackPt=cms.double(15)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
process.selpfMet.AcopCut = cms.untracked.double(100000)
process.selpfMet.MtMax = cms.untracked.double(120)
process.selpfMet.MuonTrig= cms.untracked.vstring("HLT_Mu11","HLT_Mu9")
#valid for 2010runA, for run B HLT_Mu9 is prescaled,need use HLT_Mu11



process.exoticaHSCPSkimPath=cms.Path(process.selectPfMetWMuNus+process.exoticaHSCPSeq)

process.exoticaHSCPOutputModule.fileName = cms.untracked.string('EXOHSCP.root')

process.endPath = cms.EndPath(process.exoticaHSCPOutputModule)





