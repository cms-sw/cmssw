# post-processors for reco Muon track validation in FullSim and FastSim
#
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Validation.RecoMuon.NewPostProcessor_RecoMuonValidator_cff import *

NEWpostProcessorMuonTrack = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/MuonTrack/*"),
    efficiency = cms.vstring(
        "effic_vs_eta 'Efficiency vs #eta' num_assoSimToReco_eta num_simul_eta",
        "effic_vs_pt 'Efficiency vs p_{T}' num_assoSimToReco_pT num_simul_pT",
        "effic_vs_hit 'Efficiency vs number of Hits' num_assoSimToReco_hit num_simul_hit",
        "effic_vs_phi 'Efficiency vs #phi' num_assoSimToReco_phi num_simul_phi",
        "effic_vs_dxy 'Efficiency vs dxy' num_assoSimToReco_dxy num_simul_dxy",
        "effic_vs_dz 'Efficiency vs dz' num_assoSimToReco_dz num_simul_dz",
        "effic_vs_pu 'Efficiency vs number of pile-up interactions' num_assoSimToReco_pu num_simul_pu",
        "effic_vs_Rpos 'Efficiency vs production Radius' num_assoSimToReco_Rpos num_simul_Rpos",
        "effic_vs_Zpos 'Efficiency vs production Z position' num_assoSimToReco_Zpos num_simul_Zpos",

        "fakerate_vs_eta 'Fake rate vs #eta' num_assoRecoToSim_eta num_reco_eta fake",
        "fakerate_vs_pt 'Fake rate vs p_{T}' num_assoRecoToSim_pT num_reco_pT fake",
        "fakerate_vs_hit 'Fake rate vs number of Hits' num_assoRecoToSim_hit num_reco_hit fake",
        "fakerate_vs_phi 'Fake rate vs #phi' num_assoRecoToSim_phi num_reco_phi fake",
        "fakerate_vs_dxy 'Fake rate vs dxy' num_assoRecoToSim_dxy num_reco_dxy fake",
        "fakerate_vs_dz 'Fake rate vs dz' num_assoRecoToSim_dz num_reco_dz fake",
        "fakerate_vs_pu 'Fake rate vs number of pile-up interactions' num_assoRecoToSim_pu num_reco_pu fake", 
        
        "chargeMisId_vs_eta 'Charge MisID rate vs #eta' num_chargemisid_eta num_assoSimToReco_eta", 
        "chargeMisId_vs_pt 'Charge MisID rate vs p_{T}' num_chargemisid_pT num_assoSimToReco_pT", 
        "chargeMisId_vs_phi 'Charge MisID rate vs #phi' num_chargemisid_phi num_assoSimToReco_phi",
        "chargeMisId_vs_dxy 'Charge MisID rate vs dxy' num_chargemisid_dxy num_assoSimToReco_dxy", 
        "chargeMisId_vs_dz 'Charge MisID rate vs dz' num_chargemisid_dz num_assoSimToReco_dz",
        "chargeMisId_vs_pu 'Charge MisID rate vs number of pile-up interactions' num_chargemisid_pu num_assoSimToReco_pu",
        # charge MisId determined vs number of RecHits
        "chargeMisId_vs_hit 'Charge MisID rate vs number of RecHits' num_chargemisid_hit num_assoRecoToSim_hit"
    ),
    profile = cms.untracked.vstring(
        "chi2_vs_eta_prof 'mean #chi^{2} vs #eta' chi2_vs_eta", 
        "chi2_vs_phi_prof 'mean #chi^{2} vs #phi' chi2_vs_phi", 
        "chi2_vs_nhits_prof 'mean #chi^{2} vs number of Hits' chi2_vs_nhits", 
        "nhits_vs_eta_prof 'mean number of Hits vs #eta' nhits_vs_eta",
        "nhits_vs_phi_prof 'mean number of Hits vs #phi' nhits_vs_phi", 
        "nDThits_vs_eta_prof 'mean number of DT hits vs #eta' nDThits_vs_eta",
        "nCSChits_vs_eta_prof 'mean number of CSC hits vs #eta' nCSChits_vs_eta",
        "nRPChits_vs_eta_prof 'mean number of RPC hits vs #eta' nRPChits_vs_eta",
        "nTRK_LayersWithMeas_vs_eta_prof 'mean # TRK Layers With Meas vs #eta' nTRK_LayersWithMeas_vs_eta", 
        "nPixel_LayersWithMeas_vs_eta_prof 'mean # Pixel layers With Meas vs #eta' nPixel_LayersWithMeas_vs_eta",
        "nlosthits_vs_eta_prof 'mean number of lost hits vs #eta' nlosthits_vs_eta",  
        "nhits_vs_phi_prof 'mean #hits vs #phi' nhits_vs_phi"
    ),
    resolutionLimitedFit = cms.untracked.bool(False),
    resolution = cms.vstring(
        "dxypull_vs_eta 'dxy Pull vs #eta' dxypull_vs_eta",
        "dxyres_vs_eta 'dxy Residual vs #eta' dxyres_vs_eta",
        "dxyres_vs_pt 'dxy Residual vs p_{T}' dxyres_vs_pt",
        "dzpull_vs_eta 'dz Pull vs #eta' dzpull_vs_eta",
        "dzres_vs_eta 'dz Residual vs #eta' dzres_vs_eta",
        "dzres_vs_pt 'dz Residual vs p_{T}' dzres_vs_pt",
        "phipull_vs_eta '#phi Pull vs #eta' phipull_vs_eta",
        "phipull_vs_phi '#phi Pull vs #phi' phipull_vs_phi",
        "phires_vs_eta '#phi Residual vs #eta' phires_vs_eta",
        "phires_vs_phi '#phi Residual vs #phi' phires_vs_phi",
        "phires_vs_pt '#phi Residual vs p_{T}' phires_vs_pt",
        "thetapull_vs_eta '#theta Pull vs #eta' thetapull_vs_eta",
        "thetapull_vs_phi '#theta Pull vs #phi' thetapull_vs_phi",
        "thetaCotres_vs_eta 'cot(#theta) Residual vs #eta' thetaCotres_vs_eta",
        "thetaCotres_vs_pt 'cot(#theta)) Residual vs p_{T}' thetaCotres_vs_pt",
        "ptpull_vs_eta 'p_{T} Pull vs #eta' ptpull_vs_eta",
        "ptpull_vs_phi 'p_{T} Pull vs #phi' ptpull_vs_phi",
        "ptres_vs_eta 'p_{T} Relative Residual vs #eta' ptres_vs_eta",
        "ptres_vs_phi 'p_{T} Relative Residual vs #phi' ptres_vs_phi",
        "ptres_vs_pt 'p_{T} Relative Residual vs p_{T}' ptres_vs_pt",
        "etares_vs_eta '#eta Residual vs #eta' etares_vs_eta"
    ),
    outputFileName = cms.untracked.string("")
)


NEWpostProcessorMuonTrackComp = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/MuonTrack/"),
    efficiency = cms.vstring(
    "Eff_GlbTk_Eta_mabh 'Eff_{GLB,TK} vs #eta' globalMuons/effic_vs_eta probeTrks/effic_vs_eta",
    "Eff_GlbTk_Pt_mabh 'Eff_{GLB,TK} vs p_{T}' globalMuons/effic_vs_pt probeTrks/effic_vs_pt",
    "Eff_GlbTk_Hit_mabh 'Eff_{GLB,TK} vs n Hits' globalMuons/effic_vs_hit probeTrks/effic_vs_hit",
    "Eff_GlbSta_Eta_mabh 'Eff_{GLB,STA} vs #eta' globalMuons/effic_vs_eta standAloneMuons_UpdAtVtx/effic_vs_eta",
    "Eff_GlbSta_Pt_mabh 'Eff_{GLB,STA} vs p_{T}' globalMuons/effic_vs_pt standAloneMuons_UpdAtVtx/effic_vs_pt",
    "Eff_GlbSta_Hit_mabh 'Eff_{GLB,STA} vs n Hits' globalMuons/effic_vs_hit standAloneMuons_UpdAtVtx/effic_vs_hit",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

NEWrecoMuonPostProcessors = cms.Sequence( NEWpostProcessorMuonTrack 
                                          * NEWpostProcessorMuonTrackComp 
                                          * NEWpostProcessorsRecoMuonValidator_seq )
