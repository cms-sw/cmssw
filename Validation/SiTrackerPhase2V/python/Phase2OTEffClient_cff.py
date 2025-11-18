import FWCore.ParameterSet.Config as cms

# Must match your C++ TopFolderName default/value
_topFolder = "TrackerPhase2OTL1TrackV"

# Keep outputs names unique so we never collide with anything else
# (prefix with "ClientTest_")
phase2OTEffClient = cms.EDProducer(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        f"{_topFolder}/Nominal_L1TF/FinalEfficiency",
        f"{_topFolder}/Extended_L1TF/Prompt/FinalEfficiency",
        f"{_topFolder}/Extended_L1TF/Displaced/FinalEfficiency",
    ),
    # Format per entry:
    # "<outputMEName> '<Plot Title;x-axis;y-axis>' <numeratorME> <denominatorME>"
    efficiency=cms.vstring(
        # Nominal
        "EtaEfficiency       '#eta efficiency;tracking particle #eta;Efficiency'   ../EfficiencyIngredients/match_tp_eta         ../EfficiencyIngredients/tp_eta",
        "PtEfficiency        'p_{T} efficiency;p_{T} [GeV];Efficiency'            ../EfficiencyIngredients/match_tp_pt          ../EfficiencyIngredients/tp_pt",
        "PtEfficiencyZoom    'p_{T} efficiency;p_{T} [GeV];Efficiency'            ../EfficiencyIngredients/match_tp_pt_zoom     ../EfficiencyIngredients/tp_pt_zoom",
        "d0Efficiency        'd_{0} efficiency;d_{0} [cm];Efficiency'             ../EfficiencyIngredients/match_tp_d0          ../EfficiencyIngredients/tp_d0",
        "LxyEfficiency       'L_{xy} efficiency;L_{xy} [cm];Efficiency'           ../EfficiencyIngredients/match_tp_Lxy         ../EfficiencyIngredients/tp_Lxy",
        "z0Efficiency        'z_{0} efficiency;z_{0} [cm];Efficiency'             ../EfficiencyIngredients/match_tp_z0          ../EfficiencyIngredients/tp_z0",

        # Prompt (Extended)
        # (denominators are nominal tp_*)
        "EtaEfficiency       '#eta efficiency;tracking particle #eta;Efficiency'   ../EfficiencyIngredients/match_prompt_tp_eta      ../EfficiencyIngredients/tp_eta",
        "PtEfficiency        'p_{T} efficiency;p_{T} [GeV];Efficiency'             ../EfficiencyIngredients/match_prompt_tp_pt       ../EfficiencyIngredients/tp_pt",
        "PtEfficiencyZoom    'p_{T} efficiency;p_{T} [GeV];Efficiency'             ../EfficiencyIngredients/match_prompt_tp_pt_zoom  ../EfficiencyIngredients/tp_pt_zoom",
        "d0Efficiency        'd_{0} efficiency;d_{0} [cm];Efficiency'              ../EfficiencyIngredients/match_prompt_tp_d0       ../EfficiencyIngredients/tp_d0",
        "LxyEfficiency       'L_{xy} efficiency;L_{xy} [cm];Efficiency'            ../EfficiencyIngredients/match_prompt_tp_Lxy      ../EfficiencyIngredients/tp_Lxy",
        "z0Efficiency        'z_{0} efficiency;z_{0} [cm];Efficiency'              ../EfficiencyIngredients/match_prompt_tp_z0       ../EfficiencyIngredients/tp_z0",

        # Displaced
        "EtaEfficiency       '#eta efficiency;tracking particle #eta;Efficiency'   ../EfficiencyIngredients/match_displaced_tp_eta     ../EfficiencyIngredients/tp_eta_for_dis",
        "PtEfficiency        'p_{T} efficiency;p_{T} [GeV];Efficiency'             ../EfficiencyIngredients/match_displaced_tp_pt      ../EfficiencyIngredients/tp_pt_for_dis",
        "PtEfficiencyZoom    'p_{T} efficiency;p_{T} [GeV];Efficiency'             ../EfficiencyIngredients/match_displaced_tp_pt_zoom ../EfficiencyIngredients/tp_pt_zoom_for_dis",
        "d0Efficiency        'd_{0} efficiency;d_{0} [cm];Efficiency'              ../EfficiencyIngredients/match_displaced_tp_d0      ../EfficiencyIngredients/tp_d0_for_dis",
        "LxyEfficiency       'L_{xy} efficiency;L_{xy} [cm];Efficiency'            ../EfficiencyIngredients/match_displaced_tp_Lxy     ../EfficiencyIngredients/tp_Lxy_for_dis",
        "z0Efficiency        'z_{0} efficiency;z_{0} [cm];Efficiency'              ../EfficiencyIngredients/match_displaced_tp_z0      ../EfficiencyIngredients/tp_z0_for_dis",
    ),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    verbose=cms.untracked.uint32(0)
)

_topFolderStubs = "TrackerPhase2OTStubV"

phase2OTStubEffClient = cms.EDProducer(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        f"{_topFolderStubs}/FinalEfficiency",
    ),
    # "<outputMEName> '<Title;x-axis;y-axis>' <numeratorME> <denominatorME>"
    efficiency=cms.vstring(
        "StubEfficiencyBarrel      'Stub Efficiency Barrel;tracking particle p_{T} [GeV];Efficiency'        ../EfficiencyIngredients/gen_clusters_if_stub_barrel         ../EfficiencyIngredients/gen_clusters_barrel",
        "StubEfficiencyZoomBarrel  'Stub Efficiency Zoom Barrel;tracking particle p_{T} [GeV];Efficiency'   ../EfficiencyIngredients/gen_clusters_if_stub_zoom_barrel    ../EfficiencyIngredients/gen_clusters_zoom_barrel",
        "StubEfficiencyEndcaps     'Stub Efficiency Endcaps;tracking particle p_{T} [GeV];Efficiency'       ../EfficiencyIngredients/gen_clusters_if_stub_endcaps        ../EfficiencyIngredients/gen_clusters_endcaps",
        "StubEfficiencyZoomEndcaps 'Stub Efficiency Zoom Endcaps;tracking particle p_{T} [GeV];Efficiency'  ../EfficiencyIngredients/gen_clusters_if_stub_zoom_endcaps   ../EfficiencyIngredients/gen_clusters_zoom_endcaps",
    ),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    verbose=cms.untracked.uint32(0)
)

# Expose as a Sequence so you can append with one line
phase2OTEffClientSeq = cms.Sequence(
    phase2OTEffClient      # tracks
    + phase2OTStubEffClient  # stubs
)