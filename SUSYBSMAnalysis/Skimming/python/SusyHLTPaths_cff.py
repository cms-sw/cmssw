import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# ******* (SOME OF THE) UNUSED HLT BITS **********
#"HLT1jetPE1","HLT1jetPE3","HLT1jetPE5",
#"HLTB1Jet","HLTB2Jet","HLTB3Jet","HLTB4Jet","HLTBHT",
#"HLT1Tau","HLT1Tau1MET","HLT2TauPixel"
#"HLT3jet1MET","HLT4jet1MET","HLT2jetvbfMET"
#"HLTXElectronBJet","HLTXElectron3Jet","HLTXElectron4Jet","HLTXElectronTau"
#"HLT2MuonJPsi","HLT2MuonUpsilon","HLT2MuonZ","HLTNMuonNonIso",
#"HLTB1JetMu","HLTB2JetMu","HLTB3JetMu","HLTB4JetMu","HLTBHTMu","HLTBJPsiMuMu","HLTXMuonBJet","HLTXMuonBJetSoftMuon","HLTXMuonTau","HLTXElectronMuon","HLTXElectronMuonRelaxed", 
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
# we really do not want "HLT1MuonNonIso"
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
susyHLTJetMETPath.HLTPaths = ['HLT1jet', 'HLT2jet', 'HLT3jet', 'HLT4jet', 'HLT2jet1MET', 'HLT1jet1MET', 'HLT1MET1HT', 'HLT2jetAco', 'HLT1jet1METAco', 'HLTS2jet1METNV', 'HLTS2jet1METAco', 'CandHLTSjet1MET1Aco', 'CandHLTSjet2MET1Aco', 'CandHLTS2jetAco']
susyHLTMETOnlyPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTMETOnlyPath.HLTPaths = ['HLT1MET']
susyHLTMuonPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTMuonPath.HLTPaths = ['HLT1MuonIso', 'HLT2MuonIso', 'HLT2MuonNonIso', 'HLT2MuonSameSign', 'HLTXMuonJets', 'HLTXElectronMuon', 'HLTXElectronMuonRelaxed', 'CandHLT2MuonIso']
susyHLTPhotonPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTPhotonPath.HLTPaths = ['HLT1Photon', 'HLT1PhotonRelaxed', 'HLT2Photon', 'HLT2PhotonRelaxed']
susyHLTElectronPath.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
susyHLTElectronPath.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed', 'HLT2Electron', 'HLT2ElectronRelaxed', 'HLT1EMHighEt', 'HLT1EMVeryHighEt', 'HLTXElectron1Jet', 'HLTXElectron2Jet']

