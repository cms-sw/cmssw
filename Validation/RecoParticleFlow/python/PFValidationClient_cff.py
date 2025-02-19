import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFClient_cfi import pfClient

pfJetClient = pfClient.clone()
pfJetClient.FolderNames = cms.vstring("PFJetValidation/CompWithGenJet")
pfJetClient.HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_")

pfMETClient = pfClient.clone()
pfMETClient.FolderNames = cms.vstring("PFMETValidation/CompWithGenMET")
pfMETClient.HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_")

pfElectronClient = pfClient.clone()
pfElectronClient.FolderNames = cms.vstring("PFElectronValidation/CompWithGenElectron")
pfElectronClient.HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_")
pfElectronClient.CreateEfficiencyPlots = cms.bool(True)
pfElectronClient.HistogramNamesForEfficiencyPlots = cms.vstring("pt_", "eta_", "phi_")

