# from /RelValQCD_Pt_3000_3500/CMSSW_2_1_0_pre6-RelVal-1213987236-IDEAL_V2-2nd/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO

import FWCore.ParameterSet.Config as cms

# from 

def RecoInput() : 
 return cms.Source("PoolSource",
                   debugVerbosity = cms.untracked.uint32(200),
                   debugFlag = cms.untracked.bool(True),
                   
                   fileNames = cms.untracked.vstring(
     '/store/results/icmstop/mc/BoostedTop/2008/CMSSW_2_1_8/fullsim/Zprime_ttbar_s1hadr2000w20_10TeV_reco/s1hadr2000w20_5000evt_10TeV_25528_0_9_RECO.root',
     '/store/results/icmstop/mc/BoostedTop/2008/CMSSW_2_1_8/fullsim/Zprime_ttbar_s1hadr2000w20_10TeV_reco/s1hadr2000w20_5000evt_10TeV_25528_10_19_RECO.root',
     '/store/results/icmstop/mc/BoostedTop/2008/CMSSW_2_1_8/fullsim/Zprime_ttbar_s1hadr2000w20_10TeV_reco/s1hadr2000w20_5000evt_10TeV_25528_20_29_RECO.root',
     '/store/results/icmstop/mc/BoostedTop/2008/CMSSW_2_1_8/fullsim/Zprime_ttbar_s1hadr2000w20_10TeV_reco/s1hadr2000w20_5000evt_10TeV_25528_30_39_RECO.root',
     '/store/results/icmstop/mc/BoostedTop/2008/CMSSW_2_1_8/fullsim/Zprime_ttbar_s1hadr2000w20_10TeV_reco/s1hadr2000w20_5000evt_10TeV_25528_40_49_RECO.root'
     )
                   )
