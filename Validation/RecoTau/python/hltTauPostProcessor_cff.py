import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltTauPostProcessor = DQMEDHarvester("DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/Tau/TauValidation/*"),
    efficiency = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring( # for smoother rebinning
        # Efficiency
        "Eff_vs_eta 'Efficiency vs #eta' genTauMatched_eta genTau_eta",
        "Eff_vs_phi 'Efficiency vs #phi' genTauMatched_phi genTau_phi",
        "Eff_vs_pt 'Efficiency vs p_{T}' genTauMatched_pt genTau_pt",
        "Eff_vs_mass 'Efficiency vs mass' genTauMatched_mass genTau_mass",
        # Fake rate (note: the flag 'fake' coputes 1 - ratio only for the globalEfficiency histogram)
        "Fake_vs_eta 'Fake Rate vs #eta' recoTauMatched_eta recoTau_eta fake",
        "Fake_vs_phi 'Fake Rate vs #phi' recoTauMatched_phi recoTau_phi fake",
        "Fake_vs_pt 'Fake Rate vs p_{T}' recoTauMatched_pt recoTau_pt fake",
        "Fake_vs_mass 'Fake Rate vs mass' recoTauMatched_mass recoTau_mass fake",
        # Duplicate rate
        "Dup_vs_eta 'Duplicate Rate vs #eta' recoTauMultiMatched_eta recoTau_eta",
        "Dup_vs_phi 'Duplicate Rate vs #phi' recoTauMultiMatched_phi recoTau_phi",
        "Dup_vs_pt 'Duplicate Rate vs p_{T}' recoTauMultiMatched_pt recoTau_pt",
        "Dup_vs_mass 'Duplicate Rate vs mass' recoTauMultiMatched_mass recoTau_mass",
        # Split rate
        "Split_vs_eta 'Split Rate vs #eta' genTauMultiMatched_eta genTau_eta",
        "Split_vs_phi 'Split Rate vs #phi' genTauMultiMatched_phi genTau_phi",
        "Split_vs_pt 'Split Rate vs p_{T}' genTauMultiMatched_pt genTau_pt",
        "Split_vs_mass 'Split Rate vs mass' genTauMultiMatched_mass genTau_mass",
    ),
    resolution = cms.vstring(),
    resolutionProfile = cms.untracked.vstring(),
    verbose = cms.untracked.uint32(2), 
    outputFileName = cms.untracked.string("")
)