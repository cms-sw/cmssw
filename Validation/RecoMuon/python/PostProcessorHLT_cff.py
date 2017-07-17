import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorMuonMultiTrackHLT = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/Muon/MultiTrack/*"),
    efficiency = cms.vstring(
    "effic 'Efficiency vs #eta' num_assoc(simToReco)_eta num_simul_eta",
    "efficPt 'Efficiency vs p_{T}' num_assoc(simToReco)_pT num_simul_pT",
    "effic_vs_hit 'Efficiency vs n Hits' num_assoc(simToReco)_hit num_simul_hit",
    "effic_vs_phi 'Efficiency vs #phi' num_assoc(simToReco)_phi num_simul_phi",
    "effic_vs_dxy 'Efficiency vs Dxy' num_assoc(simToReco)_dxy num_simul_dxy",
    "effic_vs_dz 'Efficiency vs Dz' num_assoc(simToReco)_dz num_simul_dz",
    "fakerate 'Fake rate vs #eta' num_assoc(recoToSim)_eta num_reco_eta fake",
    "fakeratePt 'Fake rate vs p_{T}' num_assoc(recoToSim)_pT num_reco_pT fake",
    "fakerate_vs_hit 'Fake rate vs hit' num_assoc(recoToSim)_hit num_reco_hit fake",
    "fakerate_vs_phi 'Fake rate vs phi' num_assoc(recoToSim)_phi num_reco_phi fake",
    "fakerate_vs_dxy 'Fake rate vs dxy' num_assoc(recoToSim)_dxy num_reco_dxy fake",
    "fakerate_vs_dz 'Fake rate vs dz' num_assoc(recoToSim)_dz num_reco_dz fake",

    "effic_Q05 'Efficiency vs #eta (Quality>0.5)' num_assoc(simToReco)_eta_Q05 num_simul_eta",
    "effic_Q075 'Efficiency vs #eta (Quality>0.75)' num_assoc(simToReco)_eta_Q075 num_simul_eta",
    "efficPt_Q05 'Efficiency vs p_{T} (Quality>0.5)' num_assoc(simToReco)_pT_Q05 num_simul_pT",
    "efficPt_Q075 'Efficiency vs p_{T} (Quality>0.75)' num_assoc(simToReco)_pT_Q075 num_simul_pT",
    "effic_vs_phi_Q05 'Efficiency vs #phi' num_assoc(simToReco)_phi_Q05 num_simul_phi",
    "effic_vs_phi_Q075 'Efficiency vs #phi' num_assoc(simToReco)_phi_Q075 num_simul_phi"
    ),
    resolutionLimitedFit = cms.untracked.bool(False),
    resolution = cms.vstring("cotThetares_vs_eta '#sigma(cot(#theta)) vs #eta' cotThetares_vs_eta",
                             "cotThetares_vs_pt '#sigma(cot(#theta)) vs p_{T}' cotThetares_vs_pt",
                             "dxypull_vs_eta 'd_{xy} Pull vs #eta' dxypull_vs_eta",
                             "dxyres_vs_eta '#sigma(d_{xy}) vs #eta' dxyres_vs_eta",
                             "dxyres_vs_pt '#sigma(d_{xy}) vs p_{T}' dxyres_vs_pt",
                             "dzpull_vs_eta 'd_{z} Pull vs #eta' dzpull_vs_eta",
                             "dzres_vs_eta '#sigma(d_{z}) vs #eta' dzres_vs_eta",
                             "dzres_vs_pt '#sigma(d_{z}) vs p_{T}' dzres_vs_pt",
                             "etares_vs_eta '#sigma(#eta) vs #eta' etares_vs_eta",
                             "phipull_vs_eta '#phi Pull vs #eta' phipull_vs_eta",
                             "phipull_vs_phi '#phi Pull vs #phi' phipull_vs_phi",
                             "phires_vs_eta '#sigma(#phi) vs #eta' phires_vs_eta",
                             "phires_vs_phi '#sigma(#phi) vs #phi' phires_vs_phi",
                             "phires_vs_pt '#sigma(#phi) vs p_{T}' phires_vs_pt",
                             "ptpull_vs_eta 'p_{T} Pull vs #eta' ptpull_vs_eta",
                             "ptpull_vs_phi 'p_{T} Pull vs #phi' ptpull_vs_phi",
                             "ptres_vs_eta '#sigma(p_{T}) vs #eta' ptres_vs_eta",
                             "ptres_vs_phi '#sigma(p_{T}) vs #phi' ptres_vs_phi",
                             "ptres_vs_pt '#sigma(p_{T}) vs p_{T}' ptres_vs_pt",
                             "thetapull_vs_eta '#theta Pull vs #eta' thetapull_vs_eta",
                             "thetapull_vs_phi '#theta Pull vs #phi' thetapull_vs_phi"),
    outputFileName = cms.untracked.string("")
)

postProcessorMuonMultiTrackHLTComp = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/Muon/MultiTrack/"), 
    efficiency = cms.vstring(
    "Eff_L3Tk_Eta_mabh 'Eff_{L3,TK} vs #eta' hltL3Muons/effic hltL3TkFromL2/effic",
    "Eff_L3Tk_Pt_mabh 'Eff_{L3,TK} vs p_{T}' hltL3Muons/efficPt hltL3TkFromL2/efficPt",
    "Eff_L3Tk_Hit_mabh 'Eff_{L3,TK} vs n Hits' hltL3Muons/effic_vs_hit hltL3TkFromL2/effic_vs_hit",
    "Eff_L3L2_Eta_mabh 'Eff_{L3,L2} vs #eta' hltL3Muons/effic hltL2Muons_UpdAtVtx/effic",
    "Eff_L3L2_Pt_mabh 'Eff_{L3,L2} vs p_{T}' hltL3Muons/efficPt hltL2Muons_UpdAtVtx/efficPt",
    "Eff_L3L2_Hit_mabh 'Eff_{L3,L2} vs n Hits' hltL3Muons/effic_vs_hit hltL2Muons_UpdAtVtx/effic_vs_hit",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
    )

postProcessorMuonMultiTrackHLTCompFS = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/Muon/MultiTrack/"), 
    efficiency = cms.vstring(
    "Eff_L3Tk_Eta_mabh 'Eff_{L3,TK} vs #eta' hltL3Muons_tpToL3MuonAssociationFS/effic hltL3TkFromL2_tpToL3TkMuonAssociationFS/effic",
    "Eff_L3Tk_Pt_mabh 'Eff_{L3,TK} vs p_{T}' hltL3Muons_tpToL3MuonAssociationFS/efficPt hltL3TkFromL2_tpToL3TkMuonAssociationFS/efficPt",
    "Eff_L3Tk_Hit_mabh 'Eff_{L3,TK} vs n Hits' hltL3Muons_tpToL3MuonAssociationFS/effic_vs_hit hltL3TkFromL2_tpToL3TkMuonAssociationFS/effic_vs_hit",
    "Eff_L3L2_Eta_mabh 'Eff_{L3,L2} vs #eta' hltL3Muons_tpToL3MuonAssociationFS/effic hltL2Muons_UpdatedAtVtx_tpToL2UpdMuonAssociationFS/effic",
    "Eff_L3L2_Pt_mabh 'Eff_{L3,L2} vs p_{T}' hltL3Muons_tpToL3MuonAssociationFS/efficPt hltL2Muons_UpdatedAtVtx_tpToL2UpdMuonAssociationFS/efficPt",
    "Eff_L3L2_Hit_mabh 'Eff_{L3,L2} vs n Hits' hltL3Muons_tpToL3MuonAssociationFS/effic_vs_hit hltL2Muons_UpdatedAtVtx_tpToL2UpdMuonAssociationFS/effic_vs_hit",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
    )


recoMuonPostProcessorsHLT = cms.Sequence(
    postProcessorMuonMultiTrackHLT
    *postProcessorMuonMultiTrackHLTComp
    )

recoMuonPostProcessorsHLTFastSim = cms.Sequence(
    postProcessorMuonMultiTrackHLT
    *postProcessorMuonMultiTrackHLTCompFS
    )
