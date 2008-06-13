import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# ******* (SOME OF THE) UNUSED HLT BITS **********
#"HLT1jetPE1","HLT1jetPE3","HLT1jetPE5",
#"HLT_BTagIP_Jet180","HLT_BTagIP_DoubleJet120","HLT_BTagIP_TripleJet70","HLT_BTagIP_QuadJet40","HLT_BTagIP_HT470",
#"HLT_IsoTau_MET65_Trk20","HLT_IsoTau_MET35_Trk15_L1MET","HLT_DoubleIsoTau_Trk3"
#"HLT_TripleJet60_MET60","HLT_QuadJet35_MET60","HLT_DoubleFwdJet40_MET60"
#"HLT_IsoEle10_BTagIP_Jet35","HLT_IsoEle12_TripleJet60","HLT_IsoEle12_QuadJet35","HLT_IsoEle12_IsoTau_Trk3"
#"HLT_DoubleMu3_JPsi","HLT_DoubleMu3_Upsilon","HLT_DoubleMu7_Z","HLT_TripleMu3",
#"HLT_BTagMu_Jet20_Calib","HLT_BTagMu_DoubleJet120","HLT_BTagMu_TripleJet70","HLT_BTagMu_QuadJet40","HLT_BTagMu_HT300","HLT_DoubleMu4_BJPsi","HLT_IsoMu7_BTagIP_Jet35","HLT_IsoMu7_BTagMu_Jet20","HLT_IsoMu14_IsoTau_Trk3","HLT_IsoEle8_IsoMu7","HLT_IsoEle10_Mu10_L1R", 
# JETMET Path
susyHLTJetMETPath = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# MET Only path
susyHLTMETOnlyPath = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# Muon HLT Path
#TO BE REVIEWD:
# in spreadsheet only: 1MuonIso, 2MuonIso, 2MuonNonIso, 2MuonSameSign, MuonJets, ElectronMuon, ElectronMuonRelaxed, CandHLT2MuonIso
# we really do not want "HLT_Mu15_L1Mu7"
susyHLTMuonPath = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# Photon HLT Path
susyHLTPhotonPath = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# Electron HLT Path
susyHLTElectronPath = copy.deepcopy(hltHighLevel)
susyHLTJetMETPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTJetMETPath.HLTPaths = ['HLT1jet', 'HLT_DoubleJet150', 'HLT_TripleJet85', 'HLT_QuadJet60', 'HLT_DoubleJet125_MET60', 
    'HLT_Jet180_MET60', 'HLT_MET35_HT350', 'HLT_DoubleJet125_Aco', 'HLT_Jet100_MET60_Aco', 'HLT_Jet80_Jet20_MET60_NV', 
    'HLT_DoubleJet60_MET60_Aco', 'CandHLTSjet1MET1Aco', 'CandHLTSjet2MET1Aco', 'CandHLTS2jetAco']
susyHLTMETOnlyPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTMETOnlyPath.HLTPaths = ['HLT1MET']
susyHLTMuonPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTMuonPath.HLTPaths = ['HLT_IsoMu11', 'HLT_DoubleIsoMu3', 'HLT_DoubleMu3', 'HLT_DoubleMu3_SameSign', 'HLT_IsoMu7_Jet40', 
    'HLT_IsoEle8_IsoMu7', 'HLT_IsoEle10_Mu10_L1R', 'CandHLT2MuonIso']
susyHLTPhotonPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTPhotonPath.HLTPaths = ['HLT_IsoPhoton30_L1I', 'HLT_IsoPhoton40_L1R', 'HLT_DoubleIsoPhoton20_L1I', 'HLT_DoubleIsoPhoton20_L1R']
susyHLTElectronPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTElectronPath.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R', 'HLT_EM80', 
    'HLT_EM200', 'HLT_IsoEle12_Jet40', 'HLT_IsoEle12_DoubleJet80']

