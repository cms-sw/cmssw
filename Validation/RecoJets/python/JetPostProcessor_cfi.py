import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

################# Postprocessing #########################
JetTesterPostprocessing = DQMEDHarvester('JetTesterPostProcessor',
    JetTypeRECO = cms.InputTag("ak4PFJetsCHS"),
    JetTypeMiniAOD = cms.InputTag("slimmedJets")
)  

JetPostProcessor = DQMEDHarvester("DQMGenericClient",
    subDirs=cms.untracked.vstring("JetMET/JetValidation/*"),
    efficiency = cms.vstring(
        "Eff_vs_EtaPt 'Efficiency vs #eta-p_{T}' MatchedGenEtaPt GenEtaPt",
        "Eff_vs_PhiPt 'Efficiency vs #phi-p_{T}' MatchedGenPhiPt GenPhiPt",
        "Fake_vs_EtaPt 'Fake Rate vs #eta-p_{T}' MatchedJetEtaPt JetEtaPt fake",
        "Fake_vs_PhiPt 'Fake Rate vs #phi-p_{T}' MatchedJetPhiPt JetPhiPt fake",
        "Dup_vs_EtaPt 'Duplicate Rate vs #eta-p_{T}' DuplicatesJetEtaPt JetEtaPt",
        "Dup_vs_PhiPt 'Duplicate Rate vs #phi-p_{T}' DuplicatesJetPhiPt JetPhiPt",
        "Dup_gen_vs_EtaPt 'Duplicate Gen Rate vs #eta-p_{T}' DuplicatesGenEtaPt GenEtaPt",
        "Dup_gen_vs_PhiPt 'Duplicate Gen Rate vs #phi-p_{T}' DuplicatesGenPhiPt GenPhiPt",
    ),
    efficiencyProfile = cms.untracked.vstring( # for smoother rebinning
        # Efficiency
        "Eff_vs_Eta 'Efficiency vs #eta' MatchedGenEta GenEta",
        "Eff_vs_Phi 'Efficiency vs #phi' MatchedGenPhi GenPhi",
        "Eff_vs_Pt 'Efficiency vs p_{T}' MatchedGenPt GenPt",
        "Eff_vs_Pt_B 'Efficiency vs p_{T} - Barrel' MatchedGenPt_B GenPt_B",
        "Eff_vs_Pt_E 'Efficiency vs p_{T} - Endcap' MatchedGenPt_E GenPt_E",
        "Eff_vs_Pt_F 'Efficiency vs p_{T} - Forward' MatchedGenPt_F GenPt_F",
        # Fake rate
        "Fake_vs_Eta 'Fake Rate vs #eta' MatchedJetEta JetEta fake",
        "Fake_vs_Phi 'Fake Rate vs #phi' MatchedJetPhi JetPhi fake",
        "Fake_vs_Pt 'Fake Rate vs p_{T}' MatchedJetPt JetPt fake",
        "Fake_vs_Pt_B 'Fake Rate vs p_{T} - Barrel' MatchedJetPt_B JetPt_B fake",
        "Fake_vs_Pt_E 'Fake Rate vs p_{T} - Endcap' MatchedJetPt_E JetPt_E fake",
        "Fake_vs_Pt_F 'Fake Rate vs p_{T} - Forward' MatchedJetPt_F JetPt_F fake",
        "Fake_vs_PtCorr_B 'Fake Rate vs p_{T}^{corr} - Barrel' MatchedCorrPt_B CorrJetPt_B fake",
        "Fake_vs_PtCorr_E 'Fake Rate vs p_{T}^{corr} - Endcap' MatchedCorrPt_E CorrJetPt_E fake",
        "Fake_vs_PtCorr_F 'Fake Rate vs p_{T}^{corr} - Forward' MatchedCorrPt_F CorrJetPt_F fake",
        # Duplicate rate
        "Dup_vs_Eta 'Duplicate Rate vs #eta' DuplicatesJetEta JetEta",
        "Dup_vs_Phi 'Duplicate Rate vs #phi' DuplicatesJetPhi JetPhi",
        "Dup_vs_Pt 'Duplicate Rate vs p_{T}' DuplicatesJetPt JetPt",
        "Dup_vs_Pt_B 'Duplicate Rate vs p_{T} - Barrel' DuplicatesJetPt_B JetPt_B",
        "Dup_vs_Pt_E 'Duplicate Rate vs p_{T} - Endcap' DuplicatesJetPt_E JetPt_E",
        "Dup_vs_Pt_F 'Duplicate Rate vs p_{T} - Forward' DuplicatesJetPt_F JetPt_F",
        # Duplicate gen rate
        "DupGen_vs_Eta 'Duplicate Gen Rate vs #eta' DuplicatesGenEta GenEta",
        "DupGen_vs_Phi 'Duplicate Gen Rate vs #phi' DuplicatesGenPhi GenPhi",
        "DupGen_vs_Pt 'Duplicate Gen Rate vs p_{T}' DuplicatesGenPt GenPt",
        "DupGen_vs_Pt_B 'Duplicate Gen Rate vs p_{T} - Barrel' DuplicatesGenPt_B GenPt_B",
        "DupGen_vs_Pt_E 'Duplicate Gen Rate vs p_{T} - Endcap' DuplicatesGenPt_E GenPt_E",
        "DupGen_vs_Pt_F 'Duplicate Gen Rate vs p_{T} - Forward' DuplicatesGenPt_F GenPt_F",
    ),
    resolution = cms.vstring(),
    resolutionProfile = cms.untracked.vstring(
        # Response Reco Over Gen
        "Res_RecoOverGen_GenEta 'Response RecoOverGen vs #eta^{gen}' h2d_PtRecoOverGen_GenEta rms",
        "Res_RecoOverGen_GenPhi 'Response RecoOverGen vs #phi^{gen}' h2d_PtRecoOverGen_GenPhi rms",
        "Res_RecoOverGen_GenPt 'Response RecoOverGen vs p_{T}^{gen}' h2d_PtRecoOverGen_GenPt rms",
        "Res_RecoOverGen_GenPt_B 'Response RecoOverGen vs p_{T}^{gen} - Barrel' h2d_PtRecoOverGen_GenPt_B rms",
        "Res_RecoOverGen_GenPt_E 'Response RecoOverGen vs p_{T}^{gen} - Endcap' h2d_PtRecoOverGen_GenPt_E rms",
        "Res_RecoOverGen_GenPt_F 'Response RecoOverGen vs p_{T}^{gen} - Forward' h2d_PtRecoOverGen_GenPt_F rms",
        # Response Corr Over Gen
        "Res_CorrOverGen_GenEta 'Response CorrOverGen vs #eta^{gen}' h2d_PtCorrOverGen_GenEta rms",
        "Res_CorrOverGen_GenPhi 'Response CorrOverGen vs #phi^{gen}' h2d_PtCorrOverGen_GenPhi rms",
        "Res_CorrOverGen_GenPt 'Response CorrOverGen vs p_{T}^{gen}' h2d_PtCorrOverGen_GenPt rms",
        "Res_CorrOverGen_GenPt_B 'Response CorrOverGen vs p_{T}^{gen} - Barrel' h2d_PtCorrOverGen_GenPt_B rms",
        "Res_CorrOverGen_GenPt_E 'Response CorrOverGen vs p_{T}^{gen} - Endcap' h2d_PtCorrOverGen_GenPt_E rms",
        "Res_CorrOverGen_GenPt_F 'Response CorrOverGen vs p_{T}^{gen} - Forward' h2d_PtCorrOverGen_GenPt_F rms",
        # Response Corr Over Reco
        "Res_CorrOverReco_Eta 'Response CorrOverReco vs #eta^{reco}' h2d_PtCorrOverReco_Eta rms",
        "Res_CorrOverReco_Phi 'Response CorrOverReco vs #phi^{reco}' h2d_PtCorrOverReco_Phi rms",
        "Res_CorrOverReco_Pt 'Response CorrOverReco vs p_{T}^{reco}' h2d_PtCorrOverReco_Pt rms",
        "Res_CorrOverReco_Pt_E 'Response CorrOverReco vs p_{T}^{reco} - Endcap' h2d_PtCorrOverReco_Pt_E rms",
        "Res_CorrOverReco_Pt_B 'Response CorrOverReco vs p_{T}^{reco} - Barrel' h2d_PtCorrOverReco_Pt_B rms",
        "Res_CorrOverReco_Pt_F 'Response CorrOverReco vs p_{T}^{reco} - Forward' h2d_PtCorrOverReco_Pt_F rms",
        # DeltaR distance
        "DeltaR_Eta '#DeltaR distance vs #eta^{reco}' h2d_DeltaR_Eta rms",
        "DeltaR_Phi '#DeltaR distance vs #phi^{reco}' h2d_DeltaR_Phi rms",
        "DeltaR_Pt '#DeltaR distance vs p_{T}^{reco}' h2d_DeltaR_Pt rms",
        "DeltaR_GenEta '#DeltaR distance vs #eta^{gen}' h2d_DeltaR_GenEta rms",
        "DeltaR_GenPhi '#DeltaR distance vs #phi^{gen}' h2d_DeltaR_GenPhi rms",
        "DeltaR_GenPt '#DeltaR distance vs p_{T}^{gen}' h2d_DeltaR_GenPt rms",
    ),
    verbose = cms.untracked.uint32(2), 
    outputFileName = cms.untracked.string("")
)
