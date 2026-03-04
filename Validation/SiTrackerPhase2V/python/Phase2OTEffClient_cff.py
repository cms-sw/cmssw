import FWCore.ParameterSet.Config as cms

# ============================================================================
# TRACK EFFICIENCY CLIENT
# ============================================================================
_topFolder = "TrackerPhase2OTL1TrackV"

phase2OTEffClient = cms.EDProducer(
    "DQMGenericClient",
    # FIX: Point to the folders that actually EXIST in the input file
    subDirs=cms.untracked.vstring(
        f"{_topFolder}/Nominal_L1TF",
        f"{_topFolder}/Extended_L1TF/Prompt",
        f"{_topFolder}/Extended_L1TF/Displaced",
    ),
    # FIX: Define Inputs/Outputs relative to those existing folders
    # Output: FinalEfficiency/Name
    # Input:  EfficiencyIngredients/Name
    efficiency=cms.vstring(
        # Nominal
        "FinalEfficiency/EtaEfficiency        '#eta efficiency;tracking particle #eta;Efficiency'    EfficiencyIngredients/match_tp_eta       EfficiencyIngredients/tp_eta",
        "FinalEfficiency/PtEfficiency         'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_tp_pt        EfficiencyIngredients/tp_pt",
        "FinalEfficiency/PtEfficiencyZoom     'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_tp_pt_zoom   EfficiencyIngredients/tp_pt_zoom",
        "FinalEfficiency/d0Efficiency         'd_{0} efficiency;d_{0} [cm];Efficiency'               EfficiencyIngredients/match_tp_d0        EfficiencyIngredients/tp_d0",
        "FinalEfficiency/LxyEfficiency        'L_{xy} efficiency;L_{xy} [cm];Efficiency'             EfficiencyIngredients/match_tp_Lxy       EfficiencyIngredients/tp_Lxy",
        "FinalEfficiency/z0Efficiency         'z_{0} efficiency;z_{0} [cm];Efficiency'               EfficiencyIngredients/match_tp_z0        EfficiencyIngredients/tp_z0",

        # Prompt
        # Note: If Prompt/Displaced use different ingredient names, ensure they match here.
        # Assuming the histogram names in ROOT are "match_prompt_tp_eta", etc.
        "FinalEfficiency/EtaEfficiency        '#eta efficiency;tracking particle #eta;Efficiency'    EfficiencyIngredients/match_prompt_tp_eta       EfficiencyIngredients/tp_eta",
        "FinalEfficiency/PtEfficiency         'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_prompt_tp_pt        EfficiencyIngredients/tp_pt",
        "FinalEfficiency/PtEfficiencyZoom     'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_prompt_tp_pt_zoom   EfficiencyIngredients/tp_pt_zoom",
        "FinalEfficiency/d0Efficiency         'd_{0} efficiency;d_{0} [cm];Efficiency'               EfficiencyIngredients/match_prompt_tp_d0        EfficiencyIngredients/tp_d0",
        "FinalEfficiency/LxyEfficiency        'L_{xy} efficiency;L_{xy} [cm];Efficiency'             EfficiencyIngredients/match_prompt_tp_Lxy       EfficiencyIngredients/tp_Lxy",
        "FinalEfficiency/z0Efficiency         'z_{0} efficiency;z_{0} [cm];Efficiency'               EfficiencyIngredients/match_prompt_tp_z0        EfficiencyIngredients/tp_z0",

        # Displaced
        "FinalEfficiency/EtaEfficiency        '#eta efficiency;tracking particle #eta;Efficiency'    EfficiencyIngredients/match_displaced_tp_eta       EfficiencyIngredients/tp_eta_for_dis",
        "FinalEfficiency/PtEfficiency         'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_displaced_tp_pt        EfficiencyIngredients/tp_pt_for_dis",
        "FinalEfficiency/PtEfficiencyZoom     'p_{T} efficiency;p_{T} [GeV];Efficiency'              EfficiencyIngredients/match_displaced_tp_pt_zoom   EfficiencyIngredients/tp_pt_zoom_for_dis",
        "FinalEfficiency/d0Efficiency         'd_{0} efficiency;d_{0} [cm];Efficiency'               EfficiencyIngredients/match_displaced_tp_d0        EfficiencyIngredients/tp_d0_for_dis",
        "FinalEfficiency/LxyEfficiency        'L_{xy} efficiency;L_{xy} [cm];Efficiency'             EfficiencyIngredients/match_displaced_tp_Lxy       EfficiencyIngredients/tp_Lxy_for_dis",
        "FinalEfficiency/z0Efficiency         'z_{0} efficiency;z_{0} [cm];Efficiency'               EfficiencyIngredients/match_displaced_tp_z0        EfficiencyIngredients/tp_z0_for_dis",
    ),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    verbose=cms.untracked.uint32(0)
)

# ============================================================================
# STUB EFFICIENCY CLIENT
# ============================================================================
_topFolderStubs = "TrackerPhase2OTStubV"

phase2OTStubEffClient = cms.EDProducer(
    "DQMGenericClient",
    # FIX: Point to the Top Folder (which definitely exists)
    subDirs=cms.untracked.vstring(
        f"{_topFolderStubs}",
    ),
    # FIX: Path logic: FinalEfficiency/Name vs EfficiencyIngredients/Name
    efficiency=cms.vstring(
        "FinalEfficiency/StubEfficiencyBarrel       'Stub Efficiency Barrel;tracking particle p_{T} [GeV];Efficiency'        EfficiencyIngredients/gen_clusters_if_stub_barrel         EfficiencyIngredients/gen_clusters_barrel",
        "FinalEfficiency/StubEfficiencyZoomBarrel   'Stub Efficiency Zoom Barrel;tracking particle p_{T} [GeV];Efficiency'   EfficiencyIngredients/gen_clusters_if_stub_zoom_barrel    EfficiencyIngredients/gen_clusters_zoom_barrel",
        "FinalEfficiency/StubEfficiencyEndcaps      'Stub Efficiency Endcaps;tracking particle p_{T} [GeV];Efficiency'       EfficiencyIngredients/gen_clusters_if_stub_endcaps        EfficiencyIngredients/gen_clusters_endcaps",
        "FinalEfficiency/StubEfficiencyZoomEndcaps  'Stub Efficiency Zoom Endcaps;tracking particle p_{T} [GeV];Efficiency'  EfficiencyIngredients/gen_clusters_if_stub_zoom_endcaps   EfficiencyIngredients/gen_clusters_zoom_endcaps",
    ),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    verbose=cms.untracked.uint32(0)
)

# Sequence
phase2OTEffClientSeq = cms.Sequence(
    phase2OTEffClient +
    phase2OTStubEffClient
)