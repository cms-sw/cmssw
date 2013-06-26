import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOMuSkim")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000)
)

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/596/F67BCF17-48E2-DE11-98B1-000423D94534.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/596/E432BCD7-55E2-DE11-B670-001617C3B6CC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/596/0A71AE7F-4DE2-DE11-8B2F-001D09F251CC.root'

	    )
)

#load the EventContent and Skim cff/i files for EXOMu sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOMu_EventContent_cfi')
process.load('SUSYBSMAnalysis.Skimming.EXOMu_cff')

#possible trigger modification by user, defualt HLT_Mu9 in EXOMu_cff.py
#process.exoticaMuHLT.HLTPaths = ['HLT_Mu3']

#define output file name. 
process.exoticaMuOutputModule.fileName = cms.untracked.string('EXOMu.root')#possible EventContent  modification by user
#AODSIMEventContent/AODEventContent/RECOSIMEventContent/RECOEventContent
#by uncommenting next lines.
#from Configuration.EventContent.EventContent_cff import *
#from SUSYBSMAnalysis.Skimming.EXOMu_EventContent_cfi import *
#SpecifiedEvenetContent=cms.PSet(
#    outputCommands = cms.untracked.vstring(
#      "keep *_exoticaHLTMuonFilter_*_*",
#	  "keep *_exoticaRecoMuonFilter_*_*",
#      )
#    )
#process.exoticaMuOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)
#process.exoticaMuOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)

#possible cut modification by user
#process.exoticaHLTMuonFilter.cut=  cms.string('pt > 5.0')
#process.exoticaHLTMuonFilter.minN=   cms.int32(2) 
#process.exoticaRecoMuonFilter.cut=  cms.string('pt > 15.0')

#Possible exoticaMuHLTQualitySeq or exoticaMuRecoQualitySeq selection by user

#process.exoticaMuSkimPath=cms.Path(process.exoticaMuHLTQualitySeq)
process.exoticaMuSkimPath=cms.Path(process.exoticaMuRecoQualitySeq)

process.endPath = cms.EndPath(process.exoticaMuOutputModule)
