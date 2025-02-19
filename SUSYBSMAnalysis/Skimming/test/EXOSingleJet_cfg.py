import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOSingleJetSkim")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_3_2_6/RelValTTbar_Tauola_2M_PROD/GEN-SIM-RECO/MC_31X_V8-v1/0013/F46F8CA0-D09A-DE11-9DE6-001D09F251E0.root"
	    )
)

#load the EventContent and Skim cff/i files for EXOSingleJet sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOSingleJet_EventContent_cfi')
process.load('SUSYBSMAnalysis.Skimming.EXOSingleJet_cff')


#define output file name. 
process.exoticaSingleJetOutputModule.fileName = cms.untracked.string('EXOSingleJet.root')


#user can select exoticaSingleJetHLTQualitySeq or exoticaSingleJetRecoQualitySeq
process.exoticaSingleJetSkimPath=cms.Path(process.exoticaSingleJetHLTQualitySeq)

process.endPath = cms.EndPath(process.exoticaSingleJetOutputModule)
