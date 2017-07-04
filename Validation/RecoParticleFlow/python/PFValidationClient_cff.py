import FWCore.ParameterSet.Config as cms

#from DQMOffline.PFTau.PFClient_cfi import pfClient
from DQMOffline.PFTau.PFClient_cfi import *

pfJetClient = pfClient.clone()
pfJetClient.FolderNames = cms.vstring("PFJetValidation/CompWithGenJet")
pfJetClient.HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_")
pfJetClient.CreateProfilePlots = cms.bool(True)
pfJetClient.HistogramNamesForProfilePlots = cms.vstring("delta_et_Over_et_VS_et_","delta_et_VS_et_","delta_eta_VS_et_","delta_phi_VS_et_")

pfMETClient = pfClient.clone()
pfMETClient.FolderNames = cms.vstring("PFMETValidation/CompWithGenMET")
pfMETClient.HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_")
pfMETClient.CreateProfilePlots = cms.bool(True)
pfMETClient.HistogramNamesForProfilePlots = cms.vstring("delta_et_Over_et_VS_et_","delta_et_VS_et_","delta_eta_VS_et_","delta_phi_VS_et_")

pfJetResClient = pfClientJetRes.clone()
pfJetResClient.FolderNames = cms.vstring("ElectronValidation/JetPtRes")
pfJetResClient.HistogramNames = cms.vstring("delta_et_Over_et_VS_et_", "BRdelta_et_Over_et_VS_et_", "ERdelta_et_Over_et_VS_et_")
#pfJetResClient.HistogramNames = cms.vstring("") # use this if slicingOn is true in PFJetResDQMAnalyzer
pfJetResClient.CreateEfficiencyPlots = cms.bool(False)
pfJetResClient.HistogramNamesForEfficiencyPlots = cms.vstring("pt_", "eta_", "phi_")

pfElectronClient = pfClient.clone()
pfElectronClient.FolderNames = cms.vstring("PFElectronValidation/CompWithGenElectron")
#pfElectronClient.HistogramNames = cms.vstring("delta_et_Over_et_VS_et_")
pfElectronClient.HistogramNames = cms.vstring("")
pfElectronClient.CreateEfficiencyPlots = cms.bool(True)
pfElectronClient.HistogramNamesForEfficiencyPlots = cms.vstring("pt_", "eta_", "phi_")

pfElectronClient.HistogramNamesForProjectionPlots = cms.vstring("delta_et_Over_et_VS_et_","delta_et_VS_et_","delta_eta_VS_et_","delta_phi_VS_et_")


