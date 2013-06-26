import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOHSCPSkim")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)



#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#'/store/data/Commissioning10/MinimumBias/RECO/v9/000/134/488/62B56763-D953-DF11-B4D3-00E081791851.root',
'/store/relval/CMSSW_3_2_6/RelValTTbar_Tauola_2M_PROD/GEN-SIM-RECO/MC_31X_V8-v1/0013/F46F8CA0-D09A-DE11-9DE6-001D09F251E0.root'
#   '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
   )
)

#load the EventContent and Skim cff/i files for EXOHSCP sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOHSCPSignal_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCPSignal_EventContent_cfi')

#possible event content modification.
#by uncomment below.
## from Configuration.EventContent.EventContent_cff import *
## from SUSYBSMAnalysis.Skimming.EXOHSCPSignal_EventContent_cfi import *
## SpecifiedEvenetContent=cms.PSet(
##     outputCommands = cms.untracked.vstring(
##       "keep L1GlobalTriggerReadoutRecord_*_*_*",
##       "keep recoVertexs_offlinePrimaryVertices_*_*",
##       "keep recoMuons_muons_*_*",
## 	  )
## 	)
## process.exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)

#possible change skim cut.
#process.exoticaRecoHSCPDedxFilter.trkPtMin = cms.double(20)
#process.exoticaRecoHSCPDedxFilter.dedxMin = cms.double(4)
#process.exoticaRecoHSCPMuonFilter.cut=cms.string('pt > 30.')


process.exoticaHSCPDedxSkimPath=cms.Path(process.exoticaHSCPDedxSeq)
process.exoticaHSCPMuonSkimPath=cms.Path(process.exoticaHSCPMuonSeq)

process.exoticaHSCPOutputModule.fileName = cms.untracked.string('EXOHSCPSignal.root')

process.endPath = cms.EndPath(process.exoticaHSCPOutputModule)





