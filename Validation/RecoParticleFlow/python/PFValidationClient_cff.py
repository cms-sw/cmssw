import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFClient_cfi import pfClient, pfClientJetRes
#from DQMOffline.PFTau.PFClient_cfi import *

pfJetClient = pfClient.clone(
    FolderNames = ['PFJetValidation/CompWithGenJet'],
    HistogramNames = ['delta_et_Over_et_VS_et_'],
    CreateProfilePlots = True,
    HistogramNamesForProfilePlots = ['delta_et_Over_et_VS_et_','delta_et_VS_et_','delta_eta_VS_et_','delta_phi_VS_et_']
)

pfMETClient = pfClient.clone(
    FolderNames = ['PFMETValidation/CompWithGenMET'],
    HistogramNames = ['delta_et_Over_et_VS_et_'],
    CreateProfilePlots = True,
    HistogramNamesForProfilePlots = ['delta_et_Over_et_VS_et_','delta_et_VS_et_','delta_eta_VS_et_','delta_phi_VS_et_']
)

pfJetResClient = pfClientJetRes.clone(
    FolderNames = ['PFJetResValidation/JetPtRes'],
    HistogramNames = ['delta_et_Over_et_VS_et_', 'BRdelta_et_Over_et_VS_et_', 'ERdelta_et_Over_et_VS_et_'],
    CreateEfficiencyPlots = False,
    HistogramNamesForEfficiencyPlots = ['pt_', 'eta_', 'phi_']
)

pfElectronClient = pfClient.clone(
    FolderNames = ['PFElectronValidation/CompWithGenElectron'],
    HistogramNames = [''],
    CreateEfficiencyPlots = True,
    HistogramNamesForEfficiencyPlots = ['pt_', 'eta_', 'phi_'],
    HistogramNamesForProjectionPlots = ['delta_et_Over_et_VS_et_','delta_et_VS_et_','delta_eta_VS_et_','delta_phi_VS_et_']
)
