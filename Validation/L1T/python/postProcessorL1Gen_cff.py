import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorL1Gen = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TriggerVsGen/"),
    efficiency = cms.vstring(
       "Muon_Eff_Pt 'L1 efficiency vs p_{T}' numerators_denominators/Muon_Eff_Pt_Nomin numerators_denominators/Muon_Eff_Pt_Denom", 
       "Muon_Eff_Eta 'L1 efficiency vs #eta' numerators_denominators/Muon_Eff_Eta_Nomin numerators_denominators/Muon_Eff_Eta_Denom", 
       "Muon_TurnOn_15 'L1 Turn On 15GeV' numerators_denominators/Muon_TurnOn_15_Nomin numerators_denominators/Muon_TurnOn_15_Denom", 
       "Muon_TurnOn_30 'L1 Turn On 30GeV' numerators_denominators/Muon_TurnOn_30_Nomin numerators_denominators/Muon_TurnOn_30_Denom", 
       "Egamma_Eff_Pt 'L1 efficiency vs p_{T}' numerators_denominators/Egamma_Eff_Pt_Nomin numerators_denominators/Egamma_Eff_Pt_Denom", 
       "Egamma_Eff_Eta 'L1 efficiency vs #eta' numerators_denominators/Egamma_Eff_Eta_Nomin numerators_denominators/Egamma_Eff_Eta_Denom", 
       "Egamma_TurnOn_15 'L1 Turn On 15GeV' numerators_denominators/Egamma_TurnOn_15_Nomin numerators_denominators/Egamma_TurnOn_15_Denom", 
       "Egamma_TurnOn_30 'L1 Turn On 30GeV' numerators_denominators/Egamma_TurnOn_30_Nomin numerators_denominators/Egamma_TurnOn_30_Denom", 
       "Tau_Eff_Pt 'L1 efficiency vs p_{T}' numerators_denominators/Tau_Eff_Pt_Nomin numerators_denominators/Tau_Eff_Pt_Denom", 
       "Tau_Eff_Eta 'L1 efficiency vs #eta' numerators_denominators/Tau_Eff_Eta_Nomin numerators_denominators/Tau_Eff_Eta_Denom", 
       "Tau_TurnOn_15 'L1 Turn On 15GeV' numerators_denominators/Tau_TurnOn_15_Nomin numerators_denominators/Tau_TurnOn_15_Denom", 
       "Tau_TurnOn_30 'L1 Turn On 30GeV' numerators_denominators/Tau_TurnOn_30_Nomin numerators_denominators/Tau_TurnOn_30_Denom", 
       "Jet_Eff_Pt 'L1 efficiency vs p_{T}' numerators_denominators/Jet_Eff_Pt_Nomin numerators_denominators/Jet_Eff_Pt_Denom", 
       "Jet_Eff_Eta 'L1 efficiency vs #eta' numerators_denominators/Jet_Eff_Eta_Nomin numerators_denominators/Jet_Eff_Eta_Denom", 
       "Jet_TurnOn_15 'L1 Turn On 15GeV' numerators_denominators/Jet_TurnOn_15_Nomin numerators_denominators/Jet_TurnOn_15_Denom", 
       "Jet_TurnOn_30 'L1 Turn On 30GeV' numerators_denominators/Jet_TurnOn_30_Nomin numerators_denominators/Jet_TurnOn_30_Denom", 

    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

L1GenPostProcessor = cms.Sequence(postProcessorL1Gen)
