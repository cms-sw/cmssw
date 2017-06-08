import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorL1Gen = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TriggerVsGen/"),
    efficiency = cms.vstring(
       "Muon_Eff_Pt 'L1 efficiency vs p_{T}' Muon_Eff_Pt_Nomin Muon_Eff_Pt_Denom", 
       "Muon_Eff_Eta 'L1 efficiency vs #eta' Muon_Eff_Eta_Nomin Muon_Eff_Eta_Denom", 
       "Muon_TurnOn_15 'L1 Turn On 15GeV' Muon_TurnOn_15_Nomin Muon_TurnOn_15_Denom", 
       "Muon_TurnOn_30 'L1 Turn On 30GeV' Muon_TurnOn_30_Nomin Muon_TurnOn_30_Denom", 
       "Egamma_Eff_Pt 'L1 efficiency vs p_{T}' Egamma_Eff_Pt_Nomin Egamma_Eff_Pt_Denom", 
       "Egamma_Eff_Eta 'L1 efficiency vs #eta' Egamma_Eff_Eta_Nomin Egamma_Eff_Eta_Denom", 
       "Egamma_TurnOn_15 'L1 Turn On 15GeV' Egamma_TurnOn_15_Nomin Egamma_TurnOn_15_Denom", 
       "Egamma_TurnOn_30 'L1 Turn On 30GeV' Egamma_TurnOn_30_Nomin Egamma_TurnOn_30_Denom", 
       "Tau_Eff_Pt 'L1 efficiency vs p_{T}' Tau_Eff_Pt_Nomin Tau_Eff_Pt_Denom", 
       "Tau_Eff_Eta 'L1 efficiency vs #eta' Tau_Eff_Eta_Nomin Tau_Eff_Eta_Denom", 
       "Tau_TurnOn_15 'L1 Turn On 15GeV' Tau_TurnOn_15_Nomin Tau_TurnOn_15_Denom", 
       "Tau_TurnOn_30 'L1 Turn On 30GeV' Tau_TurnOn_30_Nomin Tau_TurnOn_30_Denom", 
       "Jet_Eff_Pt 'L1 efficiency vs p_{T}' Jet_Eff_Pt_Nomin Jet_Eff_Pt_Denom", 
       "Jet_Eff_Eta 'L1 efficiency vs #eta' Jet_Eff_Eta_Nomin Jet_Eff_Eta_Denom", 
       "Jet_TurnOn_15 'L1 Turn On 15GeV' Jet_TurnOn_15_Nomin Jet_TurnOn_15_Denom", 
       "Jet_TurnOn_30 'L1 Turn On 30GeV' Jet_TurnOn_30_Nomin Jet_TurnOn_30_Denom", 

    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

L1GenPostProcessor = cms.Sequence(postProcessorL1Gen)
