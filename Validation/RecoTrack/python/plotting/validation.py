from __future__ import print_function
from __future__ import absolute_import
import os
import re
import sys
import shutil
import subprocess
import urllib

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from . import plotting
from . import html

# Mapping from releases to GlobalTags
_globalTags = {
    "CMSSW_6_2_0": {"default": "PRE_ST62_V8"},
    "CMSSW_6_2_0_SLHC15": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1"},
    "CMSSW_6_2_0_SLHC17": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1"},
    "CMSSW_6_2_0_SLHC20": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1_UPG2023SHNoTaper"},
    "CMSSW_6_2_0_SLHC22": {"UPG2023SHNoTaper": "PH2_1K_FB_V6_UPG23SHNoTaper",
                           # map 81X GReco and tilted to SHNoTaper
                           "2023GReco": "PH2_1K_FB_V6_UPG23SHNoTaper", "2023GRecoPU35": "", "2023GRecoPU140": "", "2023GRecoPU200": "",
                           "2023tilted": "PH2_1K_FB_V6_UPG23SHNoTaper", "2023tiltedPU35": "", "2023tiltedPU140": "", "2023tiltedPU200": ""},
    "CMSSW_6_2_0_SLHC26": {"LHCCRefPU140": "DES23_62_V1_LHCCRefPU140", "LHCCRefPU200": "DES23_62_V1_LHCCRefPU200",
                           # map 81X GReco and tilted to LHCCRef
                           "2023GReco": "", "2023GRecoPU35": "", "2023GRecoPU140": "DES23_62_V1_LHCCRefPU140", "2023GRecoPU200": "DES23_62_V1_LHCCRefPU200",
                           "2023tilted": "", "2023tiltedPU35": "", "2023tiltedPU140": "DES23_62_V1_LHCCRefPU140", "2023tiltedPU200": "DES23_62_V1_LHCCRefPU200"},
    "CMSSW_6_2_0_SLHC27_phase1": {"default": "DES17_62_V8_UPG17"},
    "CMSSW_7_0_0": {"default": "POSTLS170_V3", "fullsim_50ns": "POSTLS170_V4"},
    "CMSSW_7_0_0_AlcaCSA14": {"default": "POSTLS170_V5_AlcaCSA14", "fullsim_50ns": "POSTLS170_V6_AlcaCSA14"},
    "CMSSW_7_0_7_pmx": {"default": "PLS170_V7AN1", "fullsim_50ns": "PLS170_V6AN1"},
    "CMSSW_7_0_9_patch3": {"default": "PLS170_V7AN2", "fullsim_50ns": "PLS170_V6AN2"},
    "CMSSW_7_0_9_patch3_Premix": {"default": "PLS170_V7AN2", "fullsim_50ns": "PLS170_V6AN2"},
    "CMSSW_7_1_0": {"default": "POSTLS171_V15", "fullsim_50ns": "POSTLS171_V16"},
    "CMSSW_7_1_9": {"default": "POSTLS171_V17", "fullsim_50ns": "POSTLS171_V18"},
    "CMSSW_7_1_9_patch2": {"default": "POSTLS171_V17", "fullsim_50ns": "POSTLS171_V18"},
    "CMSSW_7_1_10_patch2": {"default": "MCRUN2_71_V1", "fullsim_50ns": "MCRUN2_71_V0"},
    "CMSSW_7_2_0_pre5": {"default": "POSTLS172_V3", "fullsim_50ns": "POSTLS172_V4"},
    "CMSSW_7_2_0_pre7": {"default": "PRE_LS172_V11", "fullsim_50ns": "PRE_LS172_V12"},
    "CMSSW_7_2_0_pre8": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
    "CMSSW_7_2_0": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
    "CMSSW_7_2_0_PHYS14": {"default": "PHYS14_25_V1_Phys14"},
    "CMSSW_7_2_2_patch1": {"default": "MCRUN2_72_V1", "fullsim_50ns": "MCRUN2_72_V0"},
    "CMSSW_7_2_2_patch1_Fall14DR": {"default": "MCRUN2_72_V3_71XGENSIM"},
#    "CMSSW_7_3_0_pre1": {"default": "PRE_LS172_V15", "fullsim_25ns": "PRE_LS172_V15_OldPU", "fullsim_50ns": "PRE_LS172_V16_OldPU"},
    "CMSSW_7_3_0_pre1": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
#    "CMSSW_7_3_0_pre2": {"default": "MCRUN2_73_V1_OldPU", "fullsim_50ns": "MCRUN2_73_V0_OldPU"},
    "CMSSW_7_3_0_pre2": {"default": "MCRUN2_73_V1", "fullsim_50ns": "MCRUN2_73_V0"},
    "CMSSW_7_3_0_pre3": {"default": "MCRUN2_73_V5", "fullsim_50ns": "MCRUN2_73_V4"},
    "CMSSW_7_3_0": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_3_0_71XGENSIM": {"default": "MCRUN2_73_V7_71XGENSIM"},
    "CMSSW_7_3_0_71XGENSIM_FIXGT": {"default": "MCRUN2_73_V9_71XGENSIM_FIXGT"},
    "CMSSW_7_3_1_patch1": {"default": "MCRUN2_73_V9", "fastsim": "MCRUN2_73_V7"},
    "CMSSW_7_3_1_patch1_GenSim_7113": {"default": "MCRUN2_73_V9_GenSim_7113"},
    "CMSSW_7_3_3": {"default": "MCRUN2_73_V11", "fullsim_50ns": "MCRUN2_73_V10", "fastsim": "MCRUN2_73_V13"},
    "CMSSW_7_4_0_pre1": {"default": "MCRUN2_73_V5", "fullsim_50ns": "MCRUN2_73_V4"},
    "CMSSW_7_4_0_pre2": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_4_0_pre2_73XGENSIM": {"default": "MCRUN2_73_V7_73XGENSIM_Pythia6", "fullsim_50ns": "MCRUN2_73_V6_73XGENSIM_Pythia6"},
    "CMSSW_7_4_0_pre5": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_4_0_pre5_BS": {"default": "MCRUN2_73_V9_postLS1beamspot", "fullsim_50ns": "MCRUN2_73_V8_postLS1beamspot"},
    "CMSSW_7_4_0_pre6": {"default": "MCRUN2_74_V1", "fullsim_50ns": "MCRUN2_74_V0"},
    "CMSSW_7_4_0_pre8": {"default": "MCRUN2_74_V7", "fullsim_25ns": "MCRUN2_74_V5_AsympMinGT", "fullsim_50ns": "MCRUN2_74_V4_StartupMinGT"},
    "CMSSW_7_4_0_pre8_minimal": {"default": "MCRUN2_74_V5_MinGT", "fullsim_25ns": "MCRUN2_74_V5_AsympMinGT", "fullsim_50ns": "MCRUN2_74_V4_StartupMinGT"},
    "CMSSW_7_4_0_pre8_25ns_asymptotic": {"default": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_pre8_50ns_startup":    {"default": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre8_50ns_asympref":   {"default": "MCRUN2_74_V5A_AsympMinGT"}, # for reference of 50ns asymptotic
    "CMSSW_7_4_0_pre8_50ns_asymptotic": {"default": "MCRUN2_74_V7A_AsympGT"},
    "CMSSW_7_4_0_pre8_ROOT6": {"default": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_pre8_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre8_pmx_v2": {"default": "MCRUN2_74_V7_gs_pre7", "fullsim_50ns": "MCRUN2_74_V6_gs_pre7"},
    "CMSSW_7_4_0_pre8_pmx_v3": {"default": "MCRUN2_74_V7_bis", "fullsim_50ns": "MCRUN2_74_V6_bis"},
    "CMSSW_7_4_0_pre9": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre9_ROOT6": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre9_extended": {"default": "MCRUN2_74_V7_extended"},
    "CMSSW_7_4_0": {"default": "MCRUN2_74_V7_gensim_740pre7", "fullsim_50ns": "MCRUN2_74_V6_gensim_740pre7", "fastsim": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_71XGENSIM": {"default": "MCRUN2_74_V7_GENSIM_7_1_15", "fullsim_50ns": "MCRUN2_74_V6_GENSIM_7_1_15"},
    "CMSSW_7_4_0_71XGENSIM_PU": {"default": "MCRUN2_74_V7_gs7115_puProd", "fullsim_50ns": "MCRUN2_74_V6_gs7115_puProd"},
    "CMSSW_7_4_0_71XGENSIM_PXworst": {"default": "MCRUN2_74_V7C_pxWorst_gs7115", "fullsim_50ns": "MCRUN2_74_V6A_pxWorst_gs7115"},
    "CMSSW_7_4_0_71XGENSIM_PXbest": {"default": "MCRUN2_74_V7D_pxBest_gs7115", "fullsim_50ns": "MCRUN2_74_V6B_pxBest_gs7115"},
    "CMSSW_7_4_0_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_1": {"default": "MCRUN2_74_V9_gensim_740pre7", "fullsim_50ns": "MCRUN2_74_V8_gensim_740pre7", "fastsim": "MCRUN2_74_V9"},
    "CMSSW_7_4_1_71XGENSIM": {"default": "MCRUN2_74_V9_gensim71X", "fullsim_50ns": "MCRUN2_74_V8_gensim71X"},
    "CMSSW_7_4_1_extended": {"default": "MCRUN2_74_V9_extended"},
    "CMSSW_7_4_3": {"default": "MCRUN2_74_V9", "fullsim_50ns": "MCRUN2_74_V8", "fastsim": "MCRUN2_74_V9", "fastsim_25ns": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_extended": {"default": "MCRUN2_74_V9_ext","fastsim": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_pmx": {"default": "MCRUN2_74_V9_ext", "fullsim_50ns": "MCRUN2_74_V8", "fastsim": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_patch1_unsch": {"default": "MCRUN2_74_V9_unsch", "fullsim_50ns": "MCRUN2_74_V8_unsch"},
    "CMSSW_7_4_4": {"default": "MCRUN2_74_V9_38Tbis", "fullsim_50ns": "MCRUN2_74_V8_38Tbis"},
    "CMSSW_7_4_4_0T": {"default": "MCRUN2_740TV1_0Tv2", "fullsim_50ns": "MCRUN2_740TV0_0TV2", "fullsim_25ns": "MCRUN2_740TV1_0TV2"},
    "CMSSW_7_4_6_patch6": {"default": "MCRUN2_74_V9_scheduled", "fullsim_50ns": "MCRUN2_74_V8_scheduled"},
    "CMSSW_7_4_6_patch6_unsch": {"default": "MCRUN2_74_V9", "fullsim_50ns": "MCRUN2_74_V8"},
    "CMSSW_7_4_6_patch6_noCCC": {"default": "MCRUN2_74_V9_unsch_noCCC", "fullsim_50ns": "MCRUN2_74_V8_unsch_noCCC"},
    "CMSSW_7_4_6_patch6_noCCC_v3": {"default": "MCRUN2_74_V9_unsch_noCCC_v3", "fullsim_50ns": "MCRUN2_74_V8_unsch_noCCC_v3"},
    "CMSSW_7_4_6_patch6_BS": {"default": "74X_mcRun2_asymptotic_realisticBS_v0_2015Jul24", "fullsim_50ns": "74X_mcRun2_startup_realisticBS_v0_2015Jul24PU", "fullsim_25ns": "74X_mcRun2_asymptotic_realisticBS_v0_2015Jul24PU"},
    "CMSSW_7_4_8_patch1_MT": {"default": "MCRUN2_74_V11_mulTrh", "fullsim_50ns": "MCRUN2_74_V10_mulTrh",},
    "CMSSW_7_4_12": {"default": "74X_mcRun2_asymptotic_v2", "fullsim_25ns": "74X_mcRun2_asymptotic_v2_v2", "fullsim_50ns": "74X_mcRun2_startup_v2_v2"},
    "CMSSW_7_5_0_pre1": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre2": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre3": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre4": {"default": "MCRUN2_75_V1", "fullsim_50ns": "MCRUN2_75_V0"},
    "CMSSW_7_5_0_pre5": {"default": "MCRUN2_75_V5", "fullsim_50ns": "MCRUN2_75_V4"},
    "CMSSW_7_5_0_pre6": {"default": "75X_mcRun2_asymptotic_v1", "fullsim_50ns": "75X_mcRun2_startup_v1"},
    "CMSSW_7_5_0": {"default": "75X_mcRun2_asymptotic_v1", "fullsim_50ns": "75X_mcRun2_startup_v1"},
    "CMSSW_7_5_0_71XGENSIM": {"default": "75X_mcRun2_asymptotic_v1_gs7115", "fullsim_50ns": "75X_mcRun2_startup_v1_gs7115"},
    "CMSSW_7_5_1": {"default": "75X_mcRun2_asymptotic_v3", "fullsim_50ns": "75X_mcRun2_startup_v3"},
    "CMSSW_7_5_1_71XGENSIM": {"default": "75X_mcRun2_asymptotic_v3_gs7118", "fullsim_50ns": "75X_mcRun2_startup_v3_gs7118"},
    "CMSSW_7_5_2": {"default": "75X_mcRun2_asymptotic_v5", "fullsim_50ns": "75X_mcRun2_startup_v4"},
    "CMSSW_7_6_0_pre1": {"default": "75X_mcRun2_asymptotic_v1", "fullsim_50ns": "75X_mcRun2_startup_v1"},
    "CMSSW_7_6_0_pre2": {"default": "75X_mcRun2_asymptotic_v2", "fullsim_50ns": "75X_mcRun2_startup_v2"},
    "CMSSW_7_6_0_pre3": {"default": "75X_mcRun2_asymptotic_v2", "fullsim_50ns": "75X_mcRun2_startup_v2"},
    "CMSSW_7_6_0_pre4": {"default": "76X_mcRun2_asymptotic_v1", "fullsim_50ns": "76X_mcRun2_startup_v1"},
    "CMSSW_7_6_0_pre5": {"default": "76X_mcRun2_asymptotic_v1", "fullsim_50ns": "76X_mcRun2_startup_v1"},
    "CMSSW_7_6_0_pre6": {"default": "76X_mcRun2_asymptotic_v4", "fullsim_50ns": "76X_mcRun2_startup_v4"},
    "CMSSW_7_6_0_pre7": {"default": "76X_mcRun2_asymptotic_v5", "fullsim_50ns": "76X_mcRun2_startup_v5", "fastsim": "76X_mcRun2_asymptotic_v5_resub"},
    "CMSSW_7_6_0": {"default": "76X_mcRun2_asymptotic_v11", "fullsim_50ns": "76X_mcRun2_startup_v11"},
    "CMSSW_7_6_0_71XGENSIM": {"default": "76X_mcRun2_asymptotic_v11_gs7120p2rlBS", "fullsim_50ns": "76X_mcRun2_startup_v11_gs7120p2rlBS"},
    "CMSSW_7_6_2": {"default": "76X_mcRun2_asymptotic_v12", "fullsim_50ns": "76X_mcRun2_startup_v11"},
    "CMSSW_7_6_3_patch2_0T": {"default": "76X_mcRun2_0T_v1_0Tv1GT"},
    "CMSSW_8_0_0_pre1": {"default": "76X_mcRun2_asymptotic_v11", "fullsim_50ns": "76X_mcRun2_startup_v11"},
    "CMSSW_8_0_0_pre2": {"default": "76X_mcRun2_asymptotic_v12", "fullsim_50ns": "76X_mcRun2_startup_v11"},
    "CMSSW_8_0_0_pre2_phase1": {"default": "76X_upgrade2017_design_v7"},
    "CMSSW_8_0_0_pre2_phase1_rereco": {"default": "76X_upgrade2017_design_v7_rereco"},
    "CMSSW_8_0_0_pre3_phase1": {"default": "76X_upgrade2017_design_v8"},
    "CMSSW_8_0_0_pre3_phase1_pythia8": {"default": "76X_upgrade2017_design_v8_pythia8"},
    "CMSSW_8_0_0_pre4": {"default": "76X_mcRun2_asymptotic_v13", "fullsim_50ns": "76X_mcRun2_startup_v12"},
    "CMSSW_8_0_0_pre4_phase1": {"default": "76X_upgrade2017_design_v8_UPG17"},
    "CMSSW_8_0_0_pre4_phase1_13TeV": {"default": "76X_upgrade2017_design_v8_UPG17"},
    "CMSSW_8_0_0_pre4_ecal15fb": {"default": "80X_mcRun2_asymptotic_2016EcalTune_15fb_v0_ecal15fbm1"},
    "CMSSW_8_0_0_pre4_ecal30fb": {"default": "80X_mcRun2_asymptotic_2016EcalTune_30fb_v0_ecal30fbm1"},
    "CMSSW_8_0_0_pre5": {"default": "80X_mcRun2_asymptotic_v1", "fullsim_50ns": "80X_mcRun2_startup_v1"},
    "CMSSW_8_0_0_pre5_phase1": {"default": "80X_upgrade2017_design_v1_UPG17"},
    "CMSSW_8_0_0_pre6": {"default": "80X_mcRun2_asymptotic_v4"},
    "CMSSW_8_0_0_pre6_phase1": {"default": "80X_upgrade2017_design_v3_UPG17"},
    "CMSSW_8_0_0_pre6_MT": {"default": "80X_mcRun2_asymptotic_v4_multiCoreResub"},
    "CMSSW_8_0_0": {"default": "80X_mcRun2_asymptotic_v4"},
    "CMSSW_8_0_0_patch1_phase1": {"default": "80X_upgrade2017_design_v4_UPG17"},
    "CMSSW_8_0_0_patch1_phase1_rereco": {"default": "80X_upgrade2017_design_v4_UPG17_rereco"},
    "CMSSW_8_0_0_patch2": {"default": "80X_mcRun2_asymptotic_v5_refGT"},
    "CMSSW_8_0_0_patch2_pixDynIneff": {"default": "80X_mcRun2_asymptotic_v5_2016PixDynIneff_targetGT"},
#    "CMSSW_8_0_0_patch2": {"default": "80X_mcRun2_asymptotic_v5_refGT"},
#    "CMSSW_8_0_0_patch2_pixDynIneff": {"default": "80X_mcRun2_asymptotic_v5_2016PixDynIneff_targetGT"},
    "CMSSW_8_0_0_patch2": {"default": "80X_mcRun2_asymptotic_v5_refGT_resub"},
    "CMSSW_8_0_0_patch2_pixDynIneff": {"default": "80X_mcRun2_asymptotic_v5_2016PixDynIneff_targetGT_resub"},
    "CMSSW_8_0_1": {"default": "80X_mcRun2_asymptotic_v6"},
    "CMSSW_8_0_1_71XGENSIM": {"default": "80X_mcRun2_asymptotic_v6_gs7120p2"},
    "CMSSW_8_0_1_gcc530": {"default": "80X_mcRun2_asymptotic_v6_gcc530"},
    "CMSSW_8_0_3_71XGENSIM": {"default": "80X_mcRun2_asymptotic_2016_v3_gs7120p2NewGTv3"},
    "CMSSW_8_0_3_71XGENSIM_hcal": {"default": "80X_mcRun2_asymptotic_2016_v3_gs71xNewGtHcalCust"},
    "CMSSW_8_0_3_71XGENSIM_tec": {"default": "80X_mcRun2_asymptotic_SiStripBad_TEC_CL62_for2016_v1_mc_gs7120p2TrkCoolLoop"},
    "CMSSW_8_0_5": {"default": "80X_mcRun2_asymptotic_v12_gs7120p2", "fastsim": "80X_mcRun2_asymptotic_v12"},
    "CMSSW_8_0_5_pmx": {"default": "80X_mcRun2_asymptotic_v12_gs7120p2_resub", "fastsim": "80X_mcRun2_asymptotic_v12"},
    "CMSSW_8_0_10": {"default": "80X_mcRun2_asymptotic_v14"},
    "CMSSW_8_0_10_patch1_BS": {"default": "80X_mcRun2_asymptotic_RealisticBS_25ns_13TeV2016_v1_mc_realisticBS2016"},
    "CMSSW_8_0_11": {"default": "80X_mcRun2_asymptotic_v14"},
    "CMSSW_8_0_15": {"default": "80X_mcRun2_asymptotic_v16_gs7120p2", "fastsim": "80X_mcRun2_asymptotic_v16"},
    "CMSSW_8_0_16": {"default": "80X_mcRun2_asymptotic_v16_gs7120p2", "fastsim": "80X_mcRun2_asymptotic_v16"},
    "CMSSW_8_0_16_Tranche4GT": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_gs7120p2_Tranch4GT",
                                "fastsim": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_Tranch4GT", "RelValTTbar": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_Tr4GT_resub"}, "fastsim_25ns": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_Tranch4GT"},
    "CMSSW_8_0_16_Tranche4GT_v2": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v2_Tr4GT_v2"},
    "CMSSW_8_0_16_Tranche4GT_pmx": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_gs7120p2_Tranch4GT", "fastsim": "80X_mcRun2_asymptotic_2016_TrancheIV_v0_resub"},
    "CMSSW_8_0_19_Tranche4GT_v2": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v2_Tr4GT_v2"},
    "CMSSW_8_0_20_Tranche4GT": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v4_Tr4GT_v4"},
    "CMSSW_8_0_21_Tranche4GT": {"default": "80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6"},
    "CMSSW_8_1_0_pre1": {"default": "80X_mcRun2_asymptotic_v6"},
    "CMSSW_8_1_0_pre1_phase1": {"default": "80X_upgrade2017_design_v4_UPG17", "fullsim_25ns": "80X_upgrade2017_design_v4_UPG17PU35"},
    "CMSSW_8_1_0_pre2": {"default": "80X_mcRun2_asymptotic_v10_gs810pre2", "fastsim": "80X_mcRun2_asymptotic_v10"},
    "CMSSW_8_1_0_pre2_phase1": {"default": "80X_upgrade2017_design_v9_UPG17designGT", "fullsim_25ns": "80X_upgrade2017_design_v9_UPG17PU35designGT"},
    "CMSSW_8_1_0_pre2_phase1_realGT": {"default": "80X_upgrade2017_realistic_v1_UPG17realGT", "fullsim_25ns": "80X_upgrade2017_realistic_v1_UPG17PU35realGT"},
    "CMSSW_8_1_0_pre3": {"default": "80X_mcRun2_asymptotic_v12"},
    "CMSSW_8_1_0_pre3_phase1": {"default": "80X_upgrade2017_realistic_v3_UPG17", "fullsim_25ns": "80X_upgrade2017_realistic_v3_UPG17PU35"},
    "CMSSW_8_1_0_pre4": {"default": "80X_mcRun2_asymptotic_v13"},
    "CMSSW_8_1_0_pre4_phase1": {"default": "80X_upgrade2017_realistic_v4_UPG17", "fullsim_25ns": "80X_upgrade2017_realistic_v4_UPG17PU35"},
    "CMSSW_8_1_0_pre5": {"default": "80X_mcRun2_asymptotic_v13"},
    "CMSSW_8_1_0_pre5_phase1": {"default": "80X_upgrade2017_realistic_v4_resubUPG17", "fullsim_25ns": "80X_upgrade2017_realistic_v4_resubUPG17PU35"},
    "CMSSW_8_1_0_pre6": {"default": "80X_mcRun2_asymptotic_v14"},
    "CMSSW_8_1_0_pre6_phase1": {"default": "81X_upgrade2017_realistic_v0_UPG17", "fullsim_25ns": "81X_upgrade2017_realistic_v0_UPG17PU35"},
    "CMSSW_8_1_0_pre7": {"default": "81X_mcRun2_asymptotic_v0"},
    "CMSSW_8_1_0_pre7_phase1": {"default": "81X_upgrade2017_realistic_v2_UPG17", "fullsim_25ns": "81X_upgrade2017_realistic_v2_UPG17PU35"},
    "CMSSW_8_1_0_pre7_phase1_newGT": {"default": "81X_upgrade2017_realistic_v3_UPG17newGT", "fullsim_25ns": "81X_upgrade2017_realistic_v3_UPG17PU35newGTresub"},
    "CMSSW_8_1_0_pre7_phase2": {"2023GReco": "81X_mcRun2_asymptotic_v0_2023GReco", "2023GRecoPU35": "", "2023GRecoPU140": "81X_mcRun2_asymptotic_v0_2023GRecoPU140resubmit2", "2023GRecoPU200": "81X_mcRun2_asymptotic_v0_2023GRecoPU200resubmit2",
                                "2023tilted": "81X_mcRun2_asymptotic_v0_2023tilted", "2023tiltedPU35": "", "2023tiltedPU140": "81X_mcRun2_asymptotic_v0_2023tiltedPU140resubmit2", "2023tiltedPU200": "81X_mcRun2_asymptotic_v0_2023tiltedPU200resubmit2"},
    "CMSSW_8_1_0_pre8": {"default": "81X_mcRun2_asymptotic_v1"},
    "CMSSW_8_1_0_pre8_phase1": {"default": "81X_upgrade2017_realistic_v3_UPG17", "fullsim_25ns": "81X_upgrade2017_realistic_v3_UPG17PU35"},
    "CMSSW_8_1_0_pre8_phase1_newGT": {"default": "81X_upgrade2017_realistic_v4_UPG17newGT", "fullsim_25ns": "81X_upgrade2017_realistic_v4_UPG17PU35newGT"},
    "CMSSW_8_1_0_pre8_phase1_newGT2": {"default": "81X_upgrade2017_realistic_v5_UPG17newGTset2", "fullsim_25ns": "81X_upgrade2017_realistic_v5_UPG17PU35newGTset2"},
    "CMSSW_8_1_0_pre8_phase2": {"2023GReco": "81X_mcRun2_asymptotic_v1_resub2023GReco", "2023GRecoPU35": "81X_mcRun2_asymptotic_v1_resub2023GRecoPU35", "2023GRecoPU140": "81X_mcRun2_asymptotic_v1_resub2023GRecoPU140", "2023GRecoPU200": "81X_mcRun2_asymptotic_v1_resub2023GRecoPU200",
                                "2023tilted": "81X_mcRun2_asymptotic_v1_2023tilted", "2023tiltedPU35": "81X_mcRun2_asymptotic_v1_2023tiltedPU", "2023tiltedPU140": "81X_mcRun2_asymptotic_v1_2023tiltedPU140", "2023tiltedPU200": "81X_mcRun2_asymptotic_v1_2023tiltedPU200"},
    "CMSSW_8_1_0_pre9": {"default": "81X_mcRun2_asymptotic_v2"},
    "CMSSW_8_1_0_pre9_Geant4102": {"default": "81X_mcRun2_asymptotic_v2"},
    "CMSSW_8_1_0_pre9_phase1": {"default": "81X_upgrade2017_realistic_v5_UPG17", "fullsim_25ns": "81X_upgrade2017_realistic_v5_UPG17PU35"},
    "CMSSW_8_1_0_pre9_phase1_newGT": {"default": "81X_upgrade2017_realistic_v6_UPG17newGT", "fullsim_25ns": "81X_upgrade2017_realistic_v6_UPG17PU35newGT"},
    "CMSSW_8_1_0_pre10": {"default": "81X_mcRun2_asymptotic_v5_recycle", "fullsim_25ns": "81X_mcRun2_asymptotic_v5_resub", "fastsim": "81X_mcRun2_asymptotic_v5"},
    "CMSSW_8_1_0_pre10_pmx": {"default": "81X_mcRun2_asymptotic_v5"},
    "CMSSW_8_1_0_pre10_phase1": {"default": "81X_upgrade2017_realistic_v9_UPG17resub", "fullsim_25ns": "81X_upgrade2017_realistic_v9_UPG17PU35resub"},
    "CMSSW_8_1_0_pre11": {"default": "81X_mcRun2_asymptotic_Candidate_2016_08_30_11_31_55", "fullsim_25ns": "81X_mcRun2_asymptotic_Candidate_2016_08_30_11_31_55_resub", "fastsim_25ns": "81X_mcRun2_asymptotic_Candidate_2016_08_30_11_31_55_resub2"},
    "CMSSW_8_1_0_pre11_pmx": {"default": "81X_mcRun2_asymptotic_Candidate_2016_08_30_11_31_55"},
    "CMSSW_8_1_0_pre11_phase1": {"default": "81X_upgrade2017_realistic_v9_UPG17", "fullsim_25ns": "81X_upgrade2017_realistic_v9_UPG17PU35"},
    "CMSSW_8_1_0_pre12": {"default": "81X_mcRun2_asymptotic_v8", "fullsim_25ns": "81X_mcRun2_asymptotic_v8_resub", "fastsim": "81X_mcRun2_asymptotic_v8_resub", "fastsim_25ns": "81X_mcRun2_asymptotic_v8"},
    "CMSSW_8_1_0_pre12_pmx": {"default": "81X_mcRun2_asymptotic_v8_resub", "fastsim_25ns": "81X_mcRun2_asymptotic_v8_rsub"},
    "CMSSW_8_1_0_pre12_phase1": {"default": "81X_upgrade2017_realistic_v13"},
    "CMSSW_8_1_0_pre12_phase1_newBPix": {"default": "81X_upgrade2017_realistic_newBPix_wAlign_v1_BpixGeom"},
    "CMSSW_8_1_0_pre12_phase1_newBPixFPix": {"default": "81X_upgrade2017_realistic_v13_BpixFpixGeom"},
    "CMSSW_8_1_0_pre12_phase1_newBPixFPixHCAL": {"default": "81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom"},
    "CMSSW_8_1_0_pre12_phase1_newHCAL": {"default": "81X_upgrade2017_HCALdev_v2_HcalGeom"},
    "CMSSW_8_1_0_pre12_phase1_newBPixHCAL": {"default": "81X_upgrade2017_HCALdev_v2_NewBPix_BpixHcalGeom"},
    "CMSSW_8_1_0_pre15": {"default": "81X_mcRun2_asymptotic_v11"},
    "CMSSW_8_1_0_pre15_HIP": {"default": "81X_mcRun2_asymptotic_v11_hip"},
    "CMSSW_8_1_0_pre15_PU": {"default": "81X_mcRun2_asymptotic_v11_M17"},
    "CMSSW_8_1_0_pre15_PU_HIP": {"default": "81X_mcRun2_asymptotic_v11_hipM17"},
    "CMSSW_8_1_0_pre15_phase1": {"default": "81X_upgrade2017_realistic_v17_BpixFpixHcalGeom",
                                 "Design": "81X_upgrade2017_design_IdealBS_v1_2017design", "Design_fullsim_25ns": "81X_upgrade2017_design_IdealBS_v1_design"},
    "CMSSW_8_1_0_pre16": {"default": "81X_mcRun2_asymptotic_v11"},
    "CMSSW_8_1_0_pre16_phase1": {"default": "81X_upgrade2017_realistic_v22", "Design": "81X_upgrade2017_design_IdealBS_v6"},
    "CMSSW_8_1_0": {"default": "81X_mcRun2_asymptotic_v12"},
    "CMSSW_8_1_0_phase1": {"default": "81X_upgrade2017_realistic_v26_HLT2017"},
    "CMSSW_9_0_0_pre1": {"default": "90X_mcRun2_asymptotic_v0"},
    "CMSSW_9_0_0_pre1_phase1": {"default": "90X_upgrade2017_realistic_v0",
                                "Design": "90X_upgrade2017_design_IdealBS_v0", "Design_fullsim_25ns": "90X_upgrade2017_design_IdealBS_v0_resub"},
    "CMSSW_9_0_0_pre2": {"default": "90X_mcRun2_asymptotic_v0"},
    "CMSSW_9_0_0_pre2_ROOT6": {"default": "90X_mcRun2_asymptotic_v0"},
    "CMSSW_9_0_0_pre2_phase1": {"default": "90X_upgrade2017_realistic_v0",
                                "Design": "90X_upgrade2017_design_IdealBS_v0", "Design_fullsim_25ns": "90X_upgrade2017_design_IdealBS_v0_resub"},
    "CMSSW_9_0_0_pre4": {"default": "90X_mcRun2_asymptotic_v1", "fullsim_25ns": "90X_mcRun2_asymptotic_v1_resub"},
    "CMSSW_9_0_0_pre4_phase1": {"default": "90X_upgrade2017_realistic_v6", "fullsim_25ns_PU50": "Nonexistent",
                                "Design": "90X_upgrade2017_design_IdealBS_v6"},
    "CMSSW_9_0_0_pre4_GS": {"default": "90X_mcRun2_asymptotic_v1_GSval", "fullsim_25ns": "90X_mcRun2_asymptotic_v1_GSval"},
    "CMSSW_9_0_0_pre4_phase1_ecalsrb5": {"default": "90X_upgrade2017_realistic_v6_B5"},
    "CMSSW_9_0_0_pre4_phase1_ecalsrc1": {"default": "90X_upgrade2017_realistic_v6_C1"},
    "CMSSW_9_0_0_pre4_phase1_ecalsrd7": {"default": "90X_upgrade2017_realistic_v6_D7"},
    "CMSSW_9_0_0_pre5": {"default": "90X_mcRun2_asymptotic_v4"},
    "CMSSW_9_0_0_pre5_pmx": {"default": "90X_mcRun2_asymptotic_v4", "fastsim_25ns": "90X_mcRun2_asymptotic_v4_resub"},
    "CMSSW_9_0_0_pre5_phase1": {"default": "90X_upgrade2017_realistic_v15",
                                "fullsim_25ns_PU35": "90X_upgrade2017_realistic_v15_resub", "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v15_PU50",
                                "Design": "90X_upgrade2017_design_IdealBS_v15"},
    "CMSSW_9_0_0_pre6": {"default": "90X_mcRun2_asymptotic_v4"},
    "CMSSW_9_0_0_pre6_phase1": {"default": "90X_upgrade2017_realistic_v15",
                                "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v15_PU50",
                                "Design": "90X_upgrade2017_design_IdealBS_v15"},
    "CMSSW_9_0_0": {"default": "90X_mcRun2_asymptotic_v5"},
    "CMSSW_9_0_0_phase1": {"default": "90X_upgrade2017_realistic_v20_resub",
                           "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v20_PU50_resub",
                           "Design": "90X_upgrade2017_design_IdealBS_v19_resub"},
    "CMSSW_9_0_0_gcc630": {"default": "90X_mcRun2_asymptotic_v5_gcc630"},
    "CMSSW_9_0_0_phase1_gcc630": {"default": "90X_upgrade2017_realistic_v20_gcc630"},
    "CMSSW_9_0_0_cc7": {"default": "90X_mcRun2_asymptotic_v5_cc7"},
    "CMSSW_9_0_0_phase1_cc7": {"default": "90X_upgrade2017_realistic_v20_cc7_rsb"},
    "CMSSW_9_0_2_phase1": {"default": "90X_upgrade2017_realistic_v20",
                           "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v20_PU50",
                           "Design": "90X_upgrade2017_design_IdealBS_v19"},
    "CMSSW_9_1_0_pre1": {"default": "90X_mcRun2_asymptotic_v5"},
    "CMSSW_9_1_0_pre1_phase1": {"default": "90X_upgrade2017_realistic_v20",
                                "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v20_PU50",
                                "Design": "90X_upgrade2017_design_IdealBS_v19_resub"},
    "CMSSW_9_1_0_pre2": {"default": "90X_mcRun2_asymptotic_v5"},
    "CMSSW_9_1_0_pre2_phase1": {"default": "90X_upgrade2017_realistic_v20",
                                "fullsim_25ns_PU50": "90X_upgrade2017_realistic_v20_PU50",
                                "Design": "90X_upgrade2017_design_IdealBS_v19"},
    "CMSSW_9_1_0_pre3": {"default": "91X_mcRun2_asymptotic_v2"},
    "CMSSW_9_1_0_pre3_phase1": {"default": "91X_upgrade2017_realistic_v3",
                                "fullsim_25ns_PU50": "91X_upgrade2017_realistic_v3_PU50_resub",
                                "Design": "91X_upgrade2017_design_IdealBS_v3"},
    "CMSSW_9_2_0": {"default": "91X_mcRun2_asymptotic_v3"},
    "CMSSW_9_2_0_phase1": {"default": "91X_upgrade2017_realistic_v5",
                           "fullsim_25ns_PU50": "91X_upgrade2017_realistic_v5_PU50",
                           "Design": "91X_upgrade2017_design_IdealBS_v5"},
    "CMSSW_9_2_0_phase1_PXmap": {"default": "91X_upgrade2017_realistic_v5_pixel_ideal_PXgeom"},
    "CMSSW_9_2_1_phase1": {"default": "92X_upgrade2017_realistic_v1",
                           "fullsim_25ns_PU50": "92X_upgrade2017_realistic_v1_PU50",
                           "Design": "92X_upgrade2017_design_IdealBS_v1"},
    "CMSSW_9_2_2": {"default": "91X_mcRun2_asymptotic_v3"},
    "CMSSW_9_2_2_phase1": {"default": "92X_upgrade2017_realistic_v1",
                           "fullsim_25ns_PU50": "92X_upgrade2017_realistic_v1_highPU_AVE50",
                           "Design": "92X_upgrade2017_design_IdealBS_v1"},
    "CMSSW_9_2_4_run1": {"default": "91X_mcRun1_realistic_v2"},
    "CMSSW_9_2_4": {"default": "91X_mcRun2_asymptotic_v3"},
    "CMSSW_9_2_7_phase1": {"default": "92X_upgrade2017_realistic_v7"},
    "CMSSW_9_3_0_pre1": {"default": "92X_mcRun2_asymptotic_v2"},
    "CMSSW_9_3_0_pre1_phase1": {"default": "92X_upgrade2017_realistic_v7",
                                "fullsim_25ns_PU50": "92X_upgrade2017_realistic_v7_highPU_AVE50",
                                "Design": "92X_upgrade2017_design_IdealBS_v7"},
    "CMSSW_9_3_0_pre1_run1": {"default": "92X_mcRun1_realistic_v2"},
    "CMSSW_9_3_0_pre2": {"default": "92X_mcRun2_asymptotic_v2"},
    "CMSSW_9_3_0_pre2_phase1": {"default": "92X_upgrade2017_realistic_v7",
                                "fullsim_25ns_PU50": "92X_upgrade2017_realistic_v7_highPU_AVE50_resub",
                                "Design": "92X_upgrade2017_design_IdealBS_v7"},
    "CMSSW_9_3_0_pre3": {"default": "92X_mcRun2_asymptotic_v2"},
    "CMSSW_9_3_0_pre3_phase1": {"default": "92X_upgrade2017_realistic_v10_resub",
                                "fullsim_25ns_PU50": "92X_upgrade2017_realistic_v10_highPU_AVE50_resub",
                                "Design": "92X_upgrade2017_design_IdealBS_v10_resub"},
    "CMSSW_9_3_0_pre3_phase1_pmx": {"default": "92X_upgrade2017_realistic_v10_resub2"},
    "CMSSW_9_3_0_pre4": {"default": "93X_mcRun2_asymptotic_v0"},
    "CMSSW_9_3_0_pre4_phase1": {"default": "93X_mc2017_realistic_v1",
                                "fullsim_25ns_PU50": "93X_mc2017_realistic_v1_highPU_AVE50",
                                "Design": "93X_mc2017_design_IdealBS_v1"},
    "CMSSW_9_3_0_pre5": {"default": "93X_mcRun2_asymptotic_v0"},
    "CMSSW_9_3_0_pre5_phase1": {"default": "93X_mc2017_realistic_v2",
                                "fullsim_25ns_PU50": "93X_mc2017_realistic_v2_highPU_AVE50",
                                "Design": "93X_mc2017_design_IdealBS_v2"},
    "CMSSW_9_4_0_pre1": {"default": "93X_mcRun2_asymptotic_v1"},
    "CMSSW_9_4_0_pre1_phase1": {"default": "93X_mc2017_realistic_v3",
                                "fullsim_25ns_PU50": "93X_mc2017_realistic_v3_highPU_AVE50",
                                "Design": "93X_mc2017_design_IdealBS_v3"},
    "CMSSW_9_4_0_pre2": {"default": "93X_mcRun2_asymptotic_v2"},
    "CMSSW_9_4_0_pre2_phase1": {"default": "94X_mc2017_realistic_v1",
                                "fullsim_25ns_PU50": "94X_mc2017_realistic_v1_highPU_AVE50",
                                "Design": "94X_mc2017_design_IdealBS_v0"},
    "CMSSW_9_4_0_pre3": {"default": "94X_mcRun2_asymptotic_v0"},
    "CMSSW_9_4_0_pre3_phase1": {"default": "94X_mc2017_realistic_v4",
                                "fullsim_25ns_PU50": "94X_mc2017_realistic_v4_highPU_AVE50",
                                "Design": "94X_mc2017_design_IdealBS_v4"},
    "CMSSW_9_4_0": {"default": "94X_mcRun2_asymptotic_v0"},
    "CMSSW_9_4_0_phase1": {"default": "94X_mc2017_realistic_v10",
                           "fullsim_25ns_PU50": "94X_mc2017_realistic_v10_highPU_AVE50",
                           "Design": "94X_mc2017_design_IdealBS_v5"},
    "CMSSW_10_0_0_pre1": {"default": "94X_mcRun2_asymptotic_v0"},
    "CMSSW_10_0_0_pre1_phase1": {"default": "94X_mc2017_realistic_v10",
                                 "fullsim_25ns_PU50": "94X_mc2017_realistic_v10_highPU_AVE50",
                                 "Design": "94X_mc2017_design_IdealBS_v5"},
    "CMSSW_10_0_0_pre2": {"default": "100X_mcRun2_asymptotic_v2"},
    "CMSSW_10_0_0_pre2_2017": {"default": "100X_mc2017_realistic_v1",
                               "fullsim_25ns": "100X_mc2017_realistic_v1_resub",
                               "fullsim_25ns_PU50": "100X_mc2017_realistic_v1_highPU_AVE50",
                               "Design": "100X_mc2017_design_IdealBS_v1",
                               "Design_fullsim_25ns_PU50": "Does_not_exist"}, # to avoid 2018 Design PU=50 matching to 2017 Design PU35
    "CMSSW_10_0_0_pre2_2017_pmx": {"default": "100X_mc2017_realistic_v1"},
    "CMSSW_10_0_0_pre2_2018": {"default": "100X_upgrade2018_realistic_v1",
                               "fullsim_25ns": "100X_upgrade2018_realistic_v1_resub",
                               "Design": "100X_upgrade2018_design_IdealBS_v1",
                               "Design_fullsim_25ns": "100X_upgrade2018_design_IdealBS_v1_resub"},
    "CMSSW_10_0_0_pre3": {"default": "100X_mcRun2_asymptotic_v2"},
    "CMSSW_10_0_0_pre3_2017": {"default": "100X_mc2017_realistic_v1_mahiOFF",
                               "fullsim_25ns_PU50": "100X_mc2017_realistic_v1_highPU_AVE50_mahiOFF",
                               "Design": "100X_mc2017_design_IdealBS_v1_mahiOFF"},
    "CMSSW_10_0_0_pre3_2018": {"default": "100X_upgrade2018_realistic_v4_mahiOFF",
                               "Design": "100X_upgrade2018_design_IdealBS_v3_mahiOFF"},
    "CMSSW_10_0_0_pre3_2018_pmx": {"default": "100X_upgrade2018_realistic_v4",
                                   "Design": "100X_upgrade2018_design_IdealBS_v3"},
    "CMSSW_10_0_0_pre3_2017_mahi": {"default": "100X_mc2017_realistic_v1_mahiON",
                                    "fullsim_25ns_PU50": "100X_mc2017_realistic_v1_highPU_AVE50_mahiON",
                                    "Design": "100X_mc2017_design_IdealBS_v1_mahiON"},
    "CMSSW_10_0_0_pre3_2018_mahi": {"default": "100X_upgrade2018_realistic_v4_mahiON",
                                    "Design": "100X_upgrade2018_design_IdealBS_v3_mahiON"},
    "CMSSW_10_0_0_pre3_GEANT4_2018_mahi": {"default": "100X_upgrade2018_realistic_v4_mahiON"},
    "CMSSW_10_0_0_pre3_G4VecGeom2_2018": {"default": "100X_upgrade2018_realistic_v4"},
}

_releasePostfixes = ["_AlcaCSA14", "_PHYS14", "_TEST", "_v2", "_v3", "_pmx", "_Fall14DR", "_FIXGT", "_PU", "_PXbest", "_PXworst", "_hcal", "_tec", "_71XGENSIM", "_73XGENSIM", "_BS", "_GenSim_7113", "_extended",
                     "_25ns_asymptotic", "_50ns_startup", "_50ns_asympref", "_50ns_asymptotic", "_minimal", "_0T", "_unsch", "_noCCC", "_MT", "_GS", "_rereco", "_pythia8", "_13TeV", "_realGT", "_newGT2", "_newGT", "_phase1", "_phase2", "_2017", "_2018", "_ecal15fb", "_ecal30fb", "_ecalsrb5", "_ecalsrc1", "_ecalsrd7", "_pixDynIneff", "_PXmap", "_gcc530", "_gcc630", "_cc7", "_Tranche4GT", "_newBPixFPixHCAL", "_newBPixFPix", "_newBPixHCAL", "_newBPix", "_newHCAL", "_HIP", "_run1", "_mahi"]
def _stripRelease(release):
    for pf in _releasePostfixes:
        if pf in release:
            return _stripRelease(release.replace(pf, ""))
    return release


def _getGlobalTag(sample, release):
    """Get a GlobalTag.

    Arguments:
    sample  -- Sample object
    release -- CMSSW release string
    """
    if not release in _globalTags:
        print("Release %s not found from globaltag map in validation.py" % release)
        sys.exit(1)
    gtmap = _globalTags[release]
    selectedGT = None
    if sample.hasOverrideGlobalTag():
        ogt = sample.overrideGlobalTag()
        if release in ogt:
            gtmap = _globalTags[ogt[release]]
    scenario = ""
    if sample.hasScenario():
        scenario = sample.scenario()
    sims = []
    if sample.fullsim():
        if sample.pileupEnabled():
            sim = "fullsim_"+sample.pileupType()
            sims.extend([
                sim+"_PU%d"%sample.pileupNumber(),
                sim
            ])
    elif sample.fastsim():
        sim = "fastsim"
        if sample.pileupEnabled():
            sim += "_"+sample.pileupType()
            sims.append(sim+"_PU%d"%sample.pileupNumber())
        sims.append(sim)

    selectedGT = None
    # First try with scenario+simulation
    if scenario != "":
        for sim in sims:
            selectedGT = gtmap.get(scenario+"_"+sim, None)
            if selectedGT is not None:
                break
        # Then with scenario (but only if sample specifies a scenario)
        if selectedGT is None:
            selectedGT = gtmap.get(scenario, None)
    # Then with simulation
    if selectedGT is None:
        for sim in sims:
            selectedGT = gtmap.get(sim, None)
            if selectedGT is not None:
                break
    # Finally default
    if selectedGT is None:
        selectedGT = gtmap["default"]

    if isinstance(selectedGT, dict):
        return selectedGT.get(sample.name(), selectedGT["default"])
    else:
        return selectedGT

# Mapping from release series to RelVal download URLs
_relvalUrls = {
    "6_2_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_6_2_x/",
    "7_0_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_0_x/",
    "7_1_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_1_x/",
    "7_2_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_2_x/",
    "7_3_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_3_x/",
    "7_4_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_4_x/",
    "7_5_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_5_x/",
    "7_6_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_6_x/",
    "8_0_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_8_0_x/",
    "8_1_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_8_1_x/",
    "9_0_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_9_0_x/",
    "9_1_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_9_1_x/",
    "9_2_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_9_2_x/",
    "9_3_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_9_3_x/",
    "9_4_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_9_4_x/",
    "10_0_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_10_0_x/",
}

_doElectronSamples = [
    "RelValTTbar",
    "RelValSingleElectronPt35",
    "RelValSingleElectronPt10",
]
_doConversionSamples = [
    "RelValTTbar",
    "RelValH125GGgluonfusion",
]
_doBHadronSamples = [
    "RelValTTbar"
]

def _getRelValUrl(release):
    """Get RelVal download URL for a given release."""
    version_re = re.compile("CMSSW_(?P<X>\d+)_(?P<Y>\d+)")
    m = version_re.search(release)
    if not m:
        raise Exception("Regex %s does not match to release version %s" % (version_re.pattern, release))
    version = "%s_%s_X" % (m.group("X"), m.group("Y"))
    if not version in _relvalUrls:
        print("No RelVal URL for version %s, please update _relvalUrls" % version)
        sys.exit(1)
    return _relvalUrls[version]

def _processPlotsForSample(plotterFolder, sample):
    if plotterFolder.onlyForPileup() and not sample.pileupEnabled():
        return False
    if plotterFolder.onlyForElectron() and not sample.doElectron():
        return False
    if plotterFolder.onlyForConversion() and not sample.doConversion():
        return False
    if plotterFolder.onlyForBHadron() and not sample.doBHadron():
        return False
    return True

class Sample:
    """Represents a RelVal sample."""
    def __init__(self, sample, append=None, midfix=None, putype=None, punum=0,
                 fastsim=False, fastsimCorrespondingFullsimPileup=None,
                 doElectron=None, doConversion=None, doBHadron=None,
                 version="v1", dqmVersion="0001", scenario=None, overrideGlobalTag=None, appendGlobalTag=""):
        """Constructor.

        Arguments:
        sample -- String for name of the sample

        Keyword arguments
        append  -- String for a variable name within the DWM file names, to be directly appended to sample name (e.g. "HS"; default None)
        midfix  -- String for a variable name within the DQM file names, to be appended after underscore to "sample name+append" (e.g. "13", "UP15"; default None)
        putype  -- String for pileup type (e.g. "25ns"/"50ns" for FullSim, "AVE20" for FastSim; default None)
        punum   -- String for amount of pileup (default None)
        fastsim -- Bool indicating the FastSim status (default False)
        fastsimCorrespondingFullSimPileup -- String indicating what is the FullSim pileup sample corresponding this FastSim sample. Must be set if fastsim=True and putype!=None (default None)
        doElectron -- Bool specifying if electron-specific plots should be produced (default depends on sample)
        doConversion -- Bool specifying if conversion-specific plots should be produced (default depends on sample)
        doBHadron -- Bool specifying if B-hadron-specific plots should be produced (default depends on sample)
        version -- String for dataset/DQM file version (default "v1")
        scenario -- Geometry scenario for upgrade samples (default None)
        overrideGlobalTag -- GlobalTag obtained from release information (in the form of {"release": "actualRelease"}; default None)
        appendGlobalTag -- String to append to GlobalTag (intended for one-time hacks; default "")
        """
        self._sample = sample
        self._append = append
        self._midfix = midfix
        self._putype = putype
        self._punum = punum
        self._fastsim = fastsim
        self._fastsimCorrespondingFullsimPileup = fastsimCorrespondingFullsimPileup
        self._version = version
        self._dqmVersion = dqmVersion
        self._scenario = scenario
        self._overrideGlobalTag = overrideGlobalTag
        self._appendGlobalTag = appendGlobalTag

        if doElectron is not None:
            self._doElectron = doElectron
        else:
            self._doElectron = (sample in _doElectronSamples)
        if doConversion is not None:
            self._doConversion = doConversion
        else:
            self._doConversion = (sample in _doConversionSamples)
        if doBHadron is not None:
            self._doBHadron = doBHadron
        else:
            self._doBHadron = (sample in _doBHadronSamples)

        if self._fastsim and self.hasPileup() and self._fastsimCorrespondingFullsimPileup is None:
            self._fastsimCorrespondingFullsimPileup = self._putype

    def digest(self):
        """Return a tuple uniquely identifying the sample, to be used e.g. as a key to dict"""
        return (self.name(), self.pileupNumber(), self.pileupType(), self.scenario(), self.fastsim())

    def sample(self):
        """Get the sample name"""
        return self._sample

    def name(self):
        """Get the sample name"""
        return self._sample

    def label(self):
        return self._sample

    def hasPileup(self):
        """Return True if sample has pileup (for HTML generation)"""
        return self._putype is not None

    def pileupEnabled(self):
        """Return True if pileup plots are enabled (for plot generation)"""
        return self.hasPileup()

    def pileup(self):
        """Return "PU"/"noPU" corresponding the pileup status"""
        if self.hasPileup():
            return "PU"
        else:
            return "noPU"

    def pileupType(self, release=None):
        """Return the pileup type"""
        if isinstance(self._putype, dict):
            return self._putype.get(release, self._putype["default"])
        else:
            return self._putype

    def pileupNumber(self):
        return self._punum

    def doElectron(self):
        return self._doElectron

    def doConversion(self):
        return self._doConversion

    def doBHadron(self):
        return self._doBHadron

    def version(self, release=None):
        if isinstance(self._version, dict):
            return self._version.get(release, self._version["default"])
        else:
            return self._version

    def hasScenario(self):
        return self._scenario is not None

    def scenario(self):
        return self._scenario

    def hasOverrideGlobalTag(self):
        return self._overrideGlobalTag is not None

    def overrideGlobalTag(self):
        return self._overrideGlobalTag

    def fastsim(self):
        """Return True for FastSim sample"""
        return self._fastsim

    def fullsim(self):
        """Return True for FullSim sample"""
        return not self._fastsim

    def fastsimCorrespondingFullsimPileup(self):
        return self._fastsimCorrespondingFullsimPileup

    def dirname(self, newRepository, newRelease, newSelection):
        """Return the output directory name

        Arguments:
        newRepository -- String for base directory for output files
        newRelease    -- String for CMSSW release
        newSelection  -- String for histogram selection
        """
        pileup = ""
        if self.hasPileup() and not self._fastsim:
            pileup = "_"+self._putype
        return "{newRepository}/{newRelease}/{newSelection}{pileup}/{sample}".format(
            newRepository=newRepository, newRelease=newRelease, newSelection=newSelection,
            pileup=pileup, sample=sample)

    def filename(self, newRelease):
        """Return the DQM file name

        Arguments:
        newRelease -- String for CMSSW release
        """
        pileup = ""
        fastsim = ""
        midfix = ""
        sample = self._sample
        if self._append is not None:
            midfix += self._append
        if self._midfix is not None:
            midfix += "_"+self._midfix
        if self.hasPileup():
            if self._fastsim:
                #sample = sample.replace("RelVal", "RelValFS_")
                # old style
                #pileup = "PU_"
                #midfix += "_"+self.pileupType(newRelease)
                # new style
                pileup = "PU"+self.pileupType(newRelease)+"_"
            else:
                pileup = "PU"+self.pileupType(newRelease)+"_"
        if self._fastsim:
            fastsim = "_FastSim"

        globalTag = _getGlobalTag(self, newRelease)

        fname = 'DQM_V{dqmVersion}_R000000001__{sample}{midfix}__{newrelease}-{pileup}{globaltag}{appendGlobalTag}{fastsim}-{version}__DQMIO.root'.format(
            sample=sample, midfix=midfix, newrelease=_stripRelease(newRelease),
            pileup=pileup, globaltag=globalTag, appendGlobalTag=self._appendGlobalTag, fastsim=fastsim,
            version=self.version(newRelease), dqmVersion=self._dqmVersion
        )

        return fname

    def datasetpattern(self, newRelease):
        """Return the dataset pattern

        Arguments:
        newRelease -- String for CMSSW release
        """
        pileup = ""
        fastsim = ""
        digi = ""
        if self.hasPileup():
            pileup = "-PU_"
        if self._fastsim:
            fastsim = "_FastSim-"
            digi = "DIGI-"
        else:
            fastsim = "*"
        globalTag = _getGlobalTag(self, newRelease)
        return "{sample}/{newrelease}-{pileup}{globaltag}{fastsim}{version}/GEN-SIM-{digi}RECO".format(
            sample=self._sample, newrelease=newRelease,
            pileup=pileup, globaltag=globalTag, fastsim=fastsim, digi=digi,
            version=self.version(newRelease)
            )

class Validation:
    """Base class for Tracking/Vertex validation."""
    def __init__(self, fullsimSamples, fastsimSamples, refRelease, refRepository, newRelease, newRepository, newFileModifier=None, selectionName=""):
        """Constructor.

        Arguments:
        fullsimSamples -- List of Sample objects for FullSim samples (may be empty)
        fastsimSamples -- List of Sample objects for FastSim samples (may be empty)
        refRelease    -- String for reference CMSSW release (can be None for no reference release)
        newRepository -- String for directory whete to put new files
        newRelease     -- CMSSW release to be validated
        refRepository  -- String for directory where reference root files are
        newFileModifier -- If given, a function to modify the names of the new files (function takes a string and returns a string)
        selectionName  -- If given, use this string as the selection name (appended to GlobalTag for directory names)
        """
        try:
            self._newRelease = os.environ["CMSSW_VERSION"]
        except KeyError:
            print('Error: CMSSW environment variables are not available.', file=sys.stderr)
            print('       Please run cmsenv', file=sys.stderr)
            sys.exit()

        self._fullsimSamples = fullsimSamples
        self._fastsimSamples = fastsimSamples
        self._refRelease = refRelease
        self._refRepository = refRepository
        self._newRelease = newRelease
        self._newBaseDir = os.path.join(newRepository, self._newRelease)
        self._newFileModifier = newFileModifier
        self._selectionName = selectionName

    def _getDirectoryName(self, *args, **kwargs):
        return None

    def _getSelectionName(self, *args, **kwargs):
        return self._selectionName

    def download(self):
        """Download DQM files. Requires grid certificate and asks your password for it."""
        filenames = [s.filename(self._newRelease) for s in self._fullsimSamples+self._fastsimSamples]
        if self._newFileModifier is not None:
            filenames = map(self._newFileModifier, filenames)
        filenames = [f for f in filenames if not os.path.exists(f)]
        if len(filenames) == 0:
            print("All files already downloaded")
            return

        relvalUrl = _getRelValUrl(self._newRelease)
        urls = [relvalUrl+f for f in filenames]
        certfile = os.path.join(os.environ["HOME"], ".globus", "usercert.pem")
        if not os.path.exists(certfile):
            print("Certificate file {certfile} does not exist, unable to download RelVal files from {url}".format(certfile=certfile, url=relvalUrl))
            sys.exit(1)
        keyfile = os.path.join(os.environ["HOME"], ".globus", "userkey.pem")
        if not os.path.exists(certfile):
            print("Private key file {keyfile} does not exist, unable to download RelVal files from {url}".format(keyfile=keyfile, url=relvalUrl))
            sys.exit(1)

        # curl --cert-type PEM --cert $HOME/.globus/usercert.pem --key $HOME/.globus/userkey.pem -k -O <url> -O <url>
        cmd = ["curl", "--cert-type", "PEM", "--cert", certfile, "--key", keyfile, "-k"]
        for u in urls:
            cmd.extend(["-O", u])
        print("Downloading %d files from RelVal URL %s:" % (len(filenames), relvalUrl))
        print(" "+"\n ".join(filenames))
        print("Please provide your private key pass phrase when curl asks it")
        ret = subprocess.call(cmd)
        if ret != 0:
            print("Downloading failed with exit code %d" % ret)
            sys.exit(1)

        # verify
        allFine = True
        for f in filenames:
            p = subprocess.Popen(["file", f], stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            if p.returncode != 0:
                print("file command failed with exit code %d" % p.returncode)
                sys.exit(1)
            if not "ROOT" in stdout:
                print("File {f} is not ROOT, please check the correct version, GlobalTag etc. from {url}".format(f=f, url=relvalUrl))
                allFine = False
                if os.path.exists(f):
                    os.remove(f)
        if not allFine:
            sys.exit(1)

    def createHtmlReport(self):
        return html.HtmlReport(self._newRelease, self._newBaseDir)

    def doPlots(self, plotter, plotterDrawArgs={}, limitSubFoldersOnlyTo=None, htmlReport=html.HtmlReportDummy(), doFastVsFull=True, doPhase2PU=False):
        """Create validation plots.

        Arguments:
        plotter       -- plotting.Plotter object that does the plotting

        Keyword arguments:
        plotterDrawArgs -- Dictionary for additional arguments to Plotter.draw() (default: {})
        limitSubFoldersOnlyTo   -- If not None, should be a dictionary from string to an object. The string is the name of a PlotFolder, and the object is PlotFolder-type specific to limit the subfolders to be processed. In general case the object is a list of strings, but e.g. for track iteration plots it is a function taking the algo and quality as parameters.
        htmlReport      -- Object returned by createHtmlReport(), in case HTML report generation is desired
        doFastVsFull    -- Do FastSim vs. FullSim comparison? (default: True)
        doPhase2PU      -- Do Phase2 PU 200 vs. 140 comparison (default: False)
        """
        self._plotter = plotter
        self._plotterDrawArgs = plotterDrawArgs

        # New vs. Ref
        for sample in self._fullsimSamples+self._fastsimSamples:
            # Check that the new DQM file exists
            harvestedFile = sample.filename(self._newRelease)
            if not os.path.exists(harvestedFile):
                print("Harvested file %s does not exist!" % harvestedFile)
                sys.exit(1)

            plotterInstance = plotter.readDirs(harvestedFile)
            htmlReport.beginSample(sample)
            for plotterFolder, dqmSubFolder in plotterInstance.iterFolders(limitSubFoldersOnlyTo=limitSubFoldersOnlyTo):
                if not _processPlotsForSample(plotterFolder, sample):
                    continue
                plotFiles = self._doPlots(sample, harvestedFile, plotterFolder, dqmSubFolder, htmlReport)
                htmlReport.addPlots(plotterFolder, dqmSubFolder, plotFiles)

        # Fast vs. Full
        if doFastVsFull:
            self._doFastsimFastVsFullPlots(limitSubFoldersOnlyTo, htmlReport)

        # Phase2 PU200 vs. PU 140
        if doPhase2PU:
            self._doPhase2PileupPlots(limitSubFoldersOnlyTo, htmlReport)

    def _doFastsimFastVsFullPlots(self, limitSubFoldersOnlyTo, htmlReport):
        for fast in self._fastsimSamples:
            correspondingFull = None
            for full in self._fullsimSamples:
                if fast.name() != full.name():
                    continue
                if fast.pileupEnabled():
                    if not full.pileupEnabled():
                        continue
                    if fast.fastsimCorrespondingFullsimPileup() != full.pileupType():
                        continue
                else:
                    if full.pileupEnabled():
                        continue

                if correspondingFull is None:
                    correspondingFull = full
                else:
                    raise Exception("Got multiple compatible FullSim samples for FastSim sample %s %s" % (fast.name(), fast.pileup()))
            if correspondingFull is None:
                print("WARNING: Did not find compatible FullSim sample for FastSim sample %s %s, omitting FastSim vs. FullSim comparison" % (fast.name(), fast.pileup()))
                continue

            # If we reach here, the harvestedFile must exist
            harvestedFile = fast.filename(self._newRelease)
            plotterInstance = self._plotter.readDirs(harvestedFile)
            htmlReport.beginSample(fast, fastVsFull=True)
            for plotterFolder, dqmSubFolder in plotterInstance.iterFolders(limitSubFoldersOnlyTo=limitSubFoldersOnlyTo):
                if not _processPlotsForSample(plotterFolder, fast):
                    continue
                plotFiles = self._doPlotsFastFull(fast, correspondingFull, plotterFolder, dqmSubFolder, htmlReport)
                htmlReport.addPlots(plotterFolder, dqmSubFolder, plotFiles)

    def _doPhase2PileupPlots(self, limitSubFoldersOnlyTo, htmlReport):
        def _stripScenario(name):
            puindex = name.find("PU")
            if puindex < 0:
                return name
            return name[:puindex]

        pu140samples = {}
        for sample in self._fullsimSamples:
            if sample.pileupNumber() == 140:
                key = (sample.name(), _stripScenario(sample.scenario()))
                if key in pu140samples:
                    raise Exception("Duplicate entry for sample %s in scenario %s" % (sample.name(), sample.scenar()))
                pu140samples[key] = sample

        for sample in self._fullsimSamples:
            if sample.pileupNumber() != 200:
                continue
            key = (sample.name(), _stripScenario(sample.scenario()))
            if not key in pu140samples:
                continue

            sample_pu140 = pu140samples[key]

            # If we reach here, the harvestedFile must exist
            harvestedFile = sample.filename(self._newRelease)
            plotterInstance = self._plotter.readDirs(harvestedFile)
            htmlReport.beginSample(sample, pileupComparison="vs. PU140")
            for plotterFolder, dqmSubFolder in plotterInstance.iterFolders(limitSubFoldersOnlyTo=limitSubFoldersOnlyTo):
                if not _processPlotsForSample(plotterFolder, sample):
                    continue
                plotFiles = self._doPlotsPileup(sample_pu140, sample, plotterFolder, dqmSubFolder, htmlReport)
                htmlReport.addPlots(plotterFolder, dqmSubFolder, plotFiles)


    def _getRefFileAndSelection(self, sample, plotterFolder, dqmSubFolder, selectionNameBase, valname):
        if self._refRelease is None:
            return (None, "")

        refGlobalTag = _getGlobalTag(sample, self._refRelease)
        def _createRefSelection(selectionName):
            sel = refGlobalTag+selectionNameBase+selectionName
            if sample.pileupEnabled():
                refPu = sample.pileupType(self._refRelease)
                if refPu != "":
                    sel += "_"+refPu
            return sel
        refSelection = _createRefSelection(plotterFolder.getSelectionName(dqmSubFolder))

        # Construct reference directory name, and open reference file it it exists
        refValFile = None
        triedRefValFiles = []
        tmp = [self._refRepository, self._refRelease]
        if sample.fastsim():
            tmp.extend(["fastsim", self._refRelease])
        for selName in plotterFolder.getSelectionNameIterator(dqmSubFolder):
            refSel = _createRefSelection(selName)
            refdir = os.path.join(*(tmp+[refSel, sample.name()]))

            # Open reference file if it exists
            refValFilePath = os.path.join(refdir, valname)
            if os.path.exists(refValFilePath):
                refSelection = refSel
                refValFile = ROOT.TFile.Open(refValFilePath)
                break
            else:
                triedRefValFiles.append(refValFilePath)
        if refValFile is None:
            if len(triedRefValFiles) == 1:
                if plotting.verbose:
                    print("Reference file %s not found" % triedRefValFiles[0])
            else:
                if plotting.verbose:
                    print("None of the possible reference files %s not found" % ",".join(triedRefValFiles))

        return (refValFile, refSelection)

    def _doPlots(self, sample, harvestedFile, plotterFolder, dqmSubFolder, htmlReport):
        """Do the real plotting work for a given sample and DQM subfolder"""
        # Get GlobalTags
        newGlobalTag = _getGlobalTag(sample, self._newRelease)

        # Construct selection string
        selectionNameBase = "_"+sample.pileup()
        newSelection = newGlobalTag+selectionNameBase+plotterFolder.getSelectionName(dqmSubFolder)
        if sample.pileupEnabled():
            newPu = sample.pileupType(self._newRelease)
            if newPu != "":
                newSelection += "_"+newPu

        valname = "val.{sample}.root".format(sample=sample.name())

        # Construct reference file and selection string
        (refValFile, refSelection) = self._getRefFileAndSelection(sample, plotterFolder, dqmSubFolder, selectionNameBase, valname)

        # Construct new directory name
        tmp = []
        if sample.fastsim():
            tmp.extend(["fastsim", self._newRelease])
        tmp.extend([newSelection, sample.name()])
        newsubdir = os.path.join(*tmp)
        newdir = os.path.join(self._newBaseDir, newsubdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        valnameFullPath = os.path.join(newdir, valname)

        # Copy the relevant histograms to a new validation root file
        # TODO: treat the case where dqmSubFolder is empty
        newValFile = _copySubDir(harvestedFile, valnameFullPath, plotterFolder.getPossibleDQMFolders(), dqmSubFolder.subfolder if dqmSubFolder is not None else None)
        fileList = []

        # Do the plots
        if plotting.verbose:
            print("Comparing ref and new {sim} {sample} {translatedFolder}".format(
            sim="FullSim" if not sample.fastsim() else "FastSim",
            sample=sample.name(), translatedFolder=str(dqmSubFolder.translated) if dqmSubFolder is not None else ""))
        rootFiles = [refValFile, newValFile]
        legendLabels = [
            "%s, %s %s" % (sample.name(), _stripRelease(self._refRelease), refSelection) if self._refRelease is not None else "dummy",
            "%s, %s %s" % (sample.name(), _stripRelease(self._newRelease), newSelection)
        ]
        plotterFolder.create(rootFiles, legendLabels, dqmSubFolder, isPileupSample=sample.pileupEnabled())
        fileList.extend(plotterFolder.draw(directory=newdir, **self._plotterDrawArgs))
        # Copy val file only if there were plots
        if len(fileList) > 0:
            fileList.append(valnameFullPath)

        # For tables we just try them all, and see which ones succeed
        for tableCreator in plotterFolder.getTableCreators():
            htmlReport.addTable(tableCreator.create(rootFiles, legendLabels, dqmSubFolder))

        newValFile.Close()
        if refValFile is not None:
            refValFile.Close()

        if len(fileList) == 0:
            return []

        dups = _findDuplicates(fileList)
        if len(dups) > 0:
            print("Plotter produced multiple files with names", ", ".join(dups))
            print("Typically this is a naming problem in the plotter configuration")
            sys.exit(1)

        # Move plots to new directory
        print("Created plots and %s in %s" % (valname, newdir))
        return map(lambda n: n.replace(newdir, newsubdir), fileList)

    def _doPlotsFastFull(self, fastSample, fullSample, plotterFolder, dqmSubFolder, htmlReport):
        """Do the real plotting work for FastSim vs. FullSim for a given algorithm, quality flag, and sample."""
        # Get GlobalTags
        fastGlobalTag = _getGlobalTag(fastSample, self._newRelease)
        fullGlobalTag = _getGlobalTag(fullSample, self._newRelease)

        # Construct selection string
        tmp = plotterFolder.getSelectionName(dqmSubFolder)
        fastSelection = fastGlobalTag+"_"+fastSample.pileup()+tmp
        fullSelection = fullGlobalTag+"_"+fullSample.pileup()+tmp
        if fullSample.pileupEnabled():
            fullSelection += "_"+fullSample.pileupType(self._newRelease)
            fastSelection += "_"+fastSample.pileupType(self._newRelease)

        # Construct directories for FastSim, FullSim, and for the results
        fastdir = os.path.join(self._newBaseDir, "fastsim", self._newRelease, fastSelection, fastSample.name())
        fulldir = os.path.join(self._newBaseDir, fullSelection, fullSample.name())
        newsubdir = os.path.join("fastfull", self._newRelease, fastSelection, fastSample.name())
        newdir = os.path.join(self._newBaseDir, newsubdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        # Open input root files
        valname = "val.{sample}.root".format(sample=fastSample.name())
        fastValFilePath = os.path.join(fastdir, valname)
        if not os.path.exists(fastValFilePath) and plotting.verbose:
            print("FastSim file %s not found" % fastValFilePath)
        fullValFilePath = os.path.join(fulldir, valname)
        if not os.path.exists(fullValFilePath) and plotting.verbose:
            print("FullSim file %s not found" % fullValFilePath)

        fastValFile = ROOT.TFile.Open(fastValFilePath)
        fullValFile = ROOT.TFile.Open(fullValFilePath)

        # Do plots
        if plotting.verbose:
            print("Comparing FullSim and FastSim {sample} {translatedFolder}".format(
            sample=fastSample.name(), translatedFolder=str(dqmSubFolder.translated) if dqmSubFolder is not None else ""))
        rootFiles = [fullValFile, fastValFile]
        legendLabels = [
            "FullSim %s, %s %s" % (fullSample.name(), _stripRelease(self._newRelease), fullSelection),
            "FastSim %s, %s %s" % (fastSample.name(), _stripRelease(self._newRelease), fastSelection),
        ]
        plotterFolder.create(rootFiles, legendLabels, dqmSubFolder, isPileupSample=fastSample.pileupEnabled(), requireAllHistograms=True)
        fileList = plotterFolder.draw(directory=newdir, **self._plotterDrawArgs)

        # For tables we just try them all, and see which ones succeed
        for tableCreator in plotterFolder.getTableCreators():
            htmlReport.addTable(tableCreator.create(rootFiles, legendLabels, dqmSubFolder))

        fullValFile.Close()
        fastValFile.Close()

        if len(fileList) == 0:
            return []

        dups = _findDuplicates(fileList)
        if len(dups) > 0:
            print("Plotter produced multiple files with names", ", ".join(dups))
            print("Typically this is a naming problem in the plotter configuration")
            sys.exit(1)

        # Move plots to new directory
        print("Created plots in %s" % (newdir))
        return map(lambda n: n.replace(newdir, newsubdir), fileList)

    def _doPlotsPileup(self, pu140Sample, pu200Sample, plotterFolder, dqmSubFolder, htmlReport):
        """Do the real plotting work for two pileup scenarios for a given algorithm, quality flag, and sample."""
        # Get GlobalTags
        pu140GlobalTag = _getGlobalTag(pu140Sample, self._newRelease)
        pu200GlobalTag = _getGlobalTag(pu200Sample, self._newRelease)

        # Construct selection string
        tmp = plotterFolder.getSelectionName(dqmSubFolder)
        pu140Selection = pu140GlobalTag+"_"+pu140Sample.pileup()+tmp+"_"+pu140Sample.pileupType(self._newRelease)
        pu200Selection = pu200GlobalTag+"_"+pu200Sample.pileup()+tmp+"_"+pu200Sample.pileupType(self._newRelease)

        # Construct directories for
        pu140dir = os.path.join(self._newBaseDir, pu140Selection, pu140Sample.name())
        pu200dir = os.path.join(self._newBaseDir, pu200Selection, pu200Sample.name())
        newsubdir = os.path.join("pileup", self._newRelease, pu200Selection, pu200Sample.name())
        newdir = os.path.join(self._newBaseDir, newsubdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        # Open input root files
        valname = "val.{sample}.root".format(sample=pu140Sample.name())
        pu140ValFilePath = os.path.join(pu140dir, valname)
        if not os.path.exists(pu140ValFilePath):
            if plotting.verbose:
                print("PU140 file %s not found" % pu140ValFilePath)
            return []
        pu200ValFilePath = os.path.join(pu200dir, valname)
        if not os.path.exists(pu200ValFilePath):
            if plotting.verbose:
                print("PU200 file %s not found" % pu200ValFilePath)
            return []

        pu140ValFile = ROOT.TFile.Open(pu140ValFilePath)
        pu200ValFile = ROOT.TFile.Open(pu200ValFilePath)

        # Do plots
        if plotting.verbose:
            print("Comparing PU140 and PU200 {sample} {translatedFolder}".format(
            sample=pu200Sample.name(), translatedFolder=str(dqmSubFolder.translated) if dqmSubFolder is not None else ""))
        rootFiles = [pu140ValFile, pu200ValFile]
        legendLabels = [
            "%s, %s %s" % (pu140Sample.name(), _stripRelease(self._newRelease), pu140Selection),
            "%s, %s %s" % (pu200Sample.name(), _stripRelease(self._newRelease), pu200Selection),
        ]
        plotterFolder.create(rootFiles, legendLabels, dqmSubFolder, isPileupSample=pu140Sample.pileupEnabled(), requireAllHistograms=True)
        fileList = plotterFolder.draw(directory=newdir, **self._plotterDrawArgs)

        # For tables we just try them all, and see which ones succeed
        for tableCreator in plotterFolder.getTableCreators():
            htmlReport.addTable(tableCreator.create(rootFiles, legendLabels, dqmSubFolder))

        pu200ValFile.Close()
        pu140ValFile.Close()

        if len(fileList) == 0:
            return []

        dups = _findDuplicates(fileList)
        if len(dups) > 0:
            print("Plotter produced multiple files with names", ", ".join(dups))
            print("Typically this is a naming problem in the plotter configuration")
            sys.exit(1)

        # Move plots to new directory
        print("Created plots in %s" % (newdir))
        return map(lambda n: n.replace(newdir, newsubdir), fileList)


def _copySubDir(oldfile, newfile, basenames, dirname):
    """Copy a subdirectory from oldfile to newfile.

    Arguments:
    oldfile   -- String for source TFile
    newfile   -- String for destination TFile
    basenames -- List of strings for base directories, first existing one is picked
    dirname   -- String for directory name under the base directory
    """
    oldf = ROOT.TFile.Open(oldfile)

    dirold = None
    for basename in basenames:
        dirold = oldf.GetDirectory(basename)
        if dirold:
            break
    if not dirold:
        raise Exception("Did not find any of %s directories from file %s" % (",".join(basenames), oldfile))
    if dirname:
        d = dirold.Get(dirname)
        if not d:
            raise Exception("Did not find directory %s under %s" % (dirname, dirold.GetPath()))
        dirold = d

    newf = ROOT.TFile.Open(newfile, "RECREATE")
    dirnew = newf
    for d in basenames[0].split("/"):
        dirnew = dirnew.mkdir(d)
    if dirname:
        dirnew = dirnew.mkdir(dirname)
    _copyDir(dirold, dirnew)

    oldf.Close()
    return newf

def _copyDir(src, dst):
    """Copy non-TTree objects from src TDirectory to dst TDirectory."""
    keys = src.GetListOfKeys()
    for key in keys:
        classname = key.GetClassName()
        cl = ROOT.TClass.GetClass(classname)
        if not cl:
            continue
        if not (cl.InheritsFrom("TTree") and cl.InheritsFrom("TDirectory")):
            dst.cd()
            obj = key.ReadObj()
            obj.Write()
            obj.Delete()

def _findDuplicates(lst):
    found = set()
    found2 = set()
    for x in lst:
        if x in found:
            found2.add(x)
        else:
            found.add(x)
    return list(found2)

class SimpleSample:
    def __init__(self, label, name, fileLegends, pileup=True, customPileupLabel=""):
        self._label = label
        self._name = name
        self._fileLegends = fileLegends
        self._pileup = pileup
        self._customPileupLabel = customPileupLabel

    def digest(self):
        # Label should be unique among the plotting run, so it serves also as the digest
        return self._label

    def label(self):
        return self._label

    def name(self):
        return self._name

    def files(self):
        return [t[0] for t in self._fileLegends]

    def legendLabels(self):
        return [t[1] for t in self._fileLegends]

    def fastsim(self):
        # No need to emulate the release validation fastsim behaviour here
        return False

    def pileupEnabled(self):
        return self._pileup

    def customPileupLabel(self):
        return self._customPileupLabel

    def doElectron(self):
        return True

    def doConversion(self):
        return True

    def doBHadron(self):
        return True

class SimpleValidation:
    def __init__(self, samples, newdir):
        self._samples = samples
        self._newdir = newdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        self._htmlReport = html.HtmlReportDummy()

    def createHtmlReport(self, validationName=""):
        if hasattr(self._htmlReport, "write"):
            raise Exception("HTML report object already created. There is probably some logic error in the calling code.")
        self._htmlReport = html.HtmlReport(validationName, self._newdir)
        return self._htmlReport

    def doPlots(self, plotters, plotterDrawArgs={}, **kwargs):
        self._plotterDrawArgs = plotterDrawArgs

        for sample in self._samples:
            self._subdirprefix = sample.label()
            self._labels = sample.legendLabels()
            self._htmlReport.beginSample(sample)

            self._openFiles = []
            for f in sample.files():
                if os.path.exists(f):
                    self._openFiles.append(ROOT.TFile.Open(f))
                else:
                    print("File %s not found (from sample %s), ignoring it" % (f, sample.name()))
                    self._openFiles.append(None)

            for plotter in plotters:
                self._doPlotsForPlotter(plotter, sample, **kwargs)

            for tf in self._openFiles:
                if tf is not None:
                    tf.Close()
            self._openFiles = []

    def _doPlotsForPlotter(self, plotter, sample, limitSubFoldersOnlyTo=None):
        plotterInstance = plotter.readDirs(*self._openFiles)
        for plotterFolder, dqmSubFolder in plotterInstance.iterFolders(limitSubFoldersOnlyTo=limitSubFoldersOnlyTo):
            if sample is not None and not _processPlotsForSample(plotterFolder, sample):
                continue
            plotFiles = self._doPlots(plotterFolder, dqmSubFolder)
            if len(plotFiles) > 0:
                self._htmlReport.addPlots(plotterFolder, dqmSubFolder, plotFiles)

    def _doPlots(self, plotterFolder, dqmSubFolder):
        plotterFolder.create(self._openFiles, self._labels, dqmSubFolder)
        newsubdir = self._subdirprefix+plotterFolder.getSelectionName(dqmSubFolder)
        newdir = os.path.join(self._newdir, newsubdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        fileList = plotterFolder.draw(directory=newdir, **self._plotterDrawArgs)

        for tableCreator in plotterFolder.getTableCreators():
            self._htmlReport.addTable(tableCreator.create(self._openFiles, self._labels, dqmSubFolder))


        if len(fileList) == 0:
            return fileList

        dups = _findDuplicates(fileList)
        if len(dups) > 0:
            print("Plotter produced multiple files with names", ", ".join(dups))
            print("Typically this is a naming problem in the plotter configuration")
            sys.exit(1)

        if self._plotterDrawArgs.get("separate", False):
            if not os.path.exists("%s/res"%newdir):
              os.makedirs("%s/res"%newdir)
            downloadables = ["index.php", "res/jquery-ui.js", "res/jquery.js", "res/style.css", "res/style.js", "res/theme.css"]
            for d in downloadables:
                if not os.path.exists("%s/%s" % (newdir,d)):
                    urllib.urlretrieve("https://raw.githubusercontent.com/musella/php-plots/master/%s"%d, "%s/%s"%(newdir,d))

        print("Created plots in %s" % newdir)
        return map(lambda n: n.replace(newdir, newsubdir), fileList)

class SeparateValidation:
    #Similar to the SimpleValidation
    #To be used only if `--separate` option is on
    def __init__(self, samples, newdir):
        self._samples = samples
        self._newdir = newdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        self._htmlReport = html.HtmlReportDummy()

    def createHtmlReport(self, validationName=""):
        if hasattr(self._htmlReport, "write"):
            raise Exception("HTML report object already created. There is probably some logic error in the calling code.")
        self._htmlReport = html.HtmlReport(validationName, self._newdir)
        return self._htmlReport

    def doPlots(self, plotters, plotterDrawArgs={}, **kwargs):
        self._plotterDrawArgs = plotterDrawArgs

        for sample in self._samples:
            self._subdirprefix = sample.label()
            self._labels = sample.legendLabels()
            self._htmlReport.beginSample(sample)

            self._openFiles = []
            for f in sample.files():
                if os.path.exists(f):
                    self._openFiles.append(ROOT.TFile.Open(f))
                else:
                    print("File %s not found (from sample %s), ignoring it" % (f, sample.name()))
                    self._openFiles.append(None)

            for plotter in plotters:
                self._doPlotsForPlotter(plotter, sample, **kwargs)

            for tf in self._openFiles:
                if tf is not None:
                    tf.Close()
            self._openFiles = []

    def _doPlotsForPlotter(self, plotter, sample, limitSubFoldersOnlyTo=None):
        plotterInstance = plotter.readDirs(*self._openFiles)
        for plotterFolder, dqmSubFolder in plotterInstance.iterFolders(limitSubFoldersOnlyTo=limitSubFoldersOnlyTo):
            if sample is not None and not _processPlotsForSample(plotterFolder, sample):
                continue
            plotFiles = self._doPlots(plotterFolder, dqmSubFolder)
            if len(plotFiles) > 0:
                self._htmlReport.addPlots(plotterFolder, dqmSubFolder, plotFiles)

    def _doPlots(self, plotterFolder, dqmSubFolder):
        plotterFolder.create(self._openFiles, self._labels, dqmSubFolder)
        newsubdir = self._subdirprefix+plotterFolder.getSelectionName(dqmSubFolder)
        newdir = os.path.join(self._newdir, newsubdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        fileList = plotterFolder.draw(directory=newdir, **self._plotterDrawArgs)

        # check if plots are produced
        if len(fileList) == 0:
            return fileList

        # check if there are duplicated plot
        dups = _findDuplicates(fileList)
        if len(dups) > 0:
            print("Plotter produced multiple files with names", ", ".join(dups))
            print("Typically this is a naming problem in the plotter configuration")
            sys.exit(1)

        linkList = []
        for f in fileList:
            if f[:f.rfind("/")] not in linkList :
                if str(f[:f.rfind("/")]) != str(newdir) :
                    linkList.append(f[:f.rfind("/")])

        for tableCreator in plotterFolder.getTableCreators():
            self._htmlReport.addTable(tableCreator.create(self._openFiles, self._labels, dqmSubFolder))

        for link in linkList :
            if not os.path.exists("%s/res"%link):
              os.makedirs("%s/res"%link)
            downloadables = ["index.php", "res/jquery-ui.js", "res/jquery.js", "res/style.css", "res/style.js", "res/theme.css"]
            for d in downloadables:
                if not os.path.exists("%s/%s" % (link,d)):
                    urllib.urlretrieve("https://raw.githubusercontent.com/rovere/php-plots/master/%s"%d, "%s/%s"%(link,d))

        print("Created separated plots in %s" % newdir)
        return map(lambda n: n.replace(newdir, newsubdir), linkList)
