import collections

from plotting import Subtract, FakeDuplicate, AggregateBins, ROC, Plot, PlotGroup, PlotFolder, Plotter
import validation
from html import PlotPurpose

########################################
#
# Per track collection plots
#
########################################

_maxEff = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]
_maxFake = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]

#_minMaxResol = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1]
_minMaxResol = [1e-5, 4e-5, 1e-4, 4e-4, 1e-3, 4e-3, 1e-2, 4e-2, 0.1, 0.4, 1]
_minMaxN = [5e-1, 5, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6, 5e7, 5e8, 5e9]

_effandfake1 = PlotGroup("effandfake1", [
    Plot("efficPt", title="Efficiency vs p_{T}", xtitle="TP p_{T} (GeV)", ytitle="efficiency vs p_{T}", xlog=True, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_pT", assoc="num_assoc(recoToSim)_pT", dup="num_duplicate_pT", reco="num_reco_pT", title="fake+duplicates vs p_{T}"),
         xtitle="track p_{T} (GeV)", ytitle="fake+duplicates rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("effic", xtitle="TP #eta", ytitle="efficiency vs #eta", title="", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_eta", assoc="num_assoc(recoToSim)_eta", dup="num_duplicate_eta", reco="num_reco_eta", title=""),
         xtitle="track #eta", ytitle="fake+duplicates rate vs #eta", ymax=_maxFake),
    Plot("effic_vs_phi", xtitle="TP #phi", ytitle="efficiency vs #phi", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_phi", assoc="num_assoc(recoToSim)_phi", dup="num_duplicate_phi", reco="num_reco_phi", title="fake+duplicates vs #phi"),
         xtitle="track #phi", ytitle="fake+duplicates rate vs #phi", ymax=_maxFake),
])

_effandfake2 = PlotGroup("effandfake2", [
    Plot("effic_vs_dxy", title="Efficiency vs dxy", xtitle="TP dxy (cm)", ytitle="efficiency vs dxy", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dxy", assoc="num_assoc(recoToSim)_dxy", dup="num_duplicate_dxy", reco="num_reco_dxy", title="fake+duplicates vs dxy"),
         xtitle="track dxy (cm)", ytitle="fake+duplicates rate vs dxy", ymax=_maxFake),
    Plot("effic_vs_dxypv", title="Efficiency vs dxy(PV)", xtitle="TP dxy(PV) (cm)", ytitle="efficiency vs dxy(PV)", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dxypv", assoc="num_assoc(recoToSim)_dxypv", dup="num_duplicate_dxypv", reco="num_reco_dxypv", title="fake+duplicates vs dxy(PV)"),
         xtitle="track dxy(PV) (cm)", ytitle="fake+duplicates rate vs dxy(PV)", ymax=_maxFake),
    Plot("effic_vs_dz", xtitle="TP dz (cm)", ytitle="Efficiency vs dz", title="", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dz", assoc="num_assoc(recoToSim)_dz", dup="num_duplicate_dz", reco="num_reco_dz", title=""),
         xtitle="track dz (cm)", ytitle="fake+duplicates rate vs dz", ymax=_maxFake),
    Plot("effic_vs_dzpv", xtitle="TP dz(PV) (cm)", ytitle="Efficiency vs dz(PV)", title="", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dz(PV)", assoc="num_assoc(recoToSim)_dzpv", dup="num_duplicate_dzpv", reco="num_reco_dzpv", title=""),
         xtitle="track dz(PV) (cm)", ytitle="fake+duplicates rate vs dz(PV)", ymax=_maxFake),
],
                         legendDy=0.09
)
_effandfake3 = PlotGroup("effandfake3", [
    Plot("effic_vs_hit", xtitle="TP hits", ytitle="efficiency vs hits", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_hit", assoc="num_assoc(recoToSim)_hit", dup="num_duplicate_hit", reco="num_reco_hit", title="fake+duplicates vs hit"),
         xtitle="track hits", ytitle="fake+duplicates rate vs hits", ymax=_maxFake),
    Plot("effic_vs_layer", xtitle="TP layers", ytitle="efficiency vs layers", xmax=25, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_layer", assoc="num_assoc(recoToSim)_layer", dup="num_duplicate_layer", reco="num_reco_layer", title="fake+duplicates vs layer"),
         xtitle="track layers", ytitle="fake+duplicates rate vs layers", ymax=_maxFake, xmax=25),
    Plot("effic_vs_pixellayer", xtitle="TP pixel layers", ytitle="efficiency vs pixel layers", title="", xmax=6, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_pixellayer", assoc="num_assoc(recoToSim)_pixellayer", dup="num_duplicate_pixellayer", reco="num_reco_pixellayer", title=""),
         xtitle="track pixel layers", ytitle="fake+duplicates rate vs pixel layers", ymax=_maxFake, xmax=6),
    Plot("effic_vs_3Dlayer", xtitle="TP 3D layers", ytitle="efficiency vs 3D layers", xmax=20, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_3Dlayer", assoc="num_assoc(recoToSim)_3Dlayer", dup="num_duplicate_3Dlayer", reco="num_reco_3Dlayer", title="fake+duplicates vs 3D layer"),
         xtitle="track 3D layers", ytitle="fake+duplicates rate vs 3D layers", ymax=_maxFake, xmax=20),
],
                         legendDy=0.09
)
_common = {"ymin": 0, "ymax": _maxEff}
_effvspos = PlotGroup("effvspos", [
    Plot("effic_vs_vertpos", xtitle="TP vert xy pos (cm)", ytitle="efficiency vs vert xy pos", **_common),
    Plot("effic_vs_zpos", xtitle="TP vert z pos (cm)", ytitle="efficiency vs vert z pos", **_common),
    Plot("effic_vs_dr", xlog=True, xtitle="min #DeltaR", ytitle="efficiency vs #DeltaR", **_common),
    Plot("fakerate_vs_dr", xlog=True, title="", xtitle="min #DeltaR", ytitle="Fake rate vs #DeltaR", ymin=0, ymax=_maxFake)
],
                         legendDy=-0.025
)

_dupandfake1 = PlotGroup("dupandfake1", [
    Plot("fakeratePt", xtitle="track p_{T} (GeV)", ytitle="fakerate vs p_{T}", xlog=True, ymax=_maxFake),
    Plot("duplicatesRate_Pt", xtitle="track p_{T} (GeV)", ytitle="duplicates rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("pileuprate_Pt", xtitle="track p_{T} (GeV)", ytitle="pileup rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("fakerate", xtitle="track #eta", ytitle="fakerate vs #eta", title="", ymax=_maxFake),
    Plot("duplicatesRate", xtitle="track #eta", ytitle="duplicates rate vs #eta", title="", ymax=_maxFake),
    Plot("pileuprate", xtitle="track #eta", ytitle="pileup rate vs #eta", title="", ymax=_maxFake),
    Plot("fakerate_vs_phi", xtitle="track #phi", ytitle="fakerate vs #phi", ymax=_maxFake),
    Plot("duplicatesRate_phi", xtitle="track #phi", ytitle="duplicates rate vs #phi", ymax=_maxFake),
    Plot("pileuprate_phi", xtitle="track #phi", ytitle="pileup rate vs #phi", ymax=_maxFake),
], ncols=3)
_dupandfake2 = PlotGroup("dupandfake2", [
    Plot("fakerate_vs_dxy", xtitle="track dxy (cm)", ytitle="fakerate vs dxy", ymax=_maxFake),
    Plot("duplicatesRate_dxy", xtitle="track dxy (cm)", ytitle="duplicates rate vs dxy", ymax=_maxFake),
    Plot("pileuprate_dxy", xtitle="track dxy (cm)", ytitle="pileup rate vs dxy", ymax=_maxFake),
    #
    Plot("fakerate_vs_dxypv", xtitle="track dxy(PV) (cm)", ytitle="fakerate vs dxy(PV)", ymax=_maxFake),
    Plot("duplicatesRate_dxypv", xtitle="track dxy(PV) (cm)", ytitle="duplicates rate vs dxy(PV)", ymax=_maxFake),
    Plot("pileuprate_dxypv", xtitle="track dxy(PV) (cm)", ytitle="pileup rate vs dxy(PV)", ymax=_maxFake),
    #
    Plot("fakerate_vs_dz", xtitle="track dz (cm)", ytitle="fakerate vs dz", title="", ymax=_maxFake),
    Plot("duplicatesRate_dz", xtitle="track dz (cm)", ytitle="duplicates rate vs dz", title="", ymax=_maxFake),
    Plot("pileuprate_dz", xtitle="track dz (cm)", ytitle="pileup rate vs dz", title="", ymax=_maxFake),
    #
    Plot("fakerate_vs_dzpv", xtitle="track dz(PV) (cm)", ytitle="fakerate vs dz(PV)", title="", ymax=_maxFake),
    Plot("duplicatesRate_dzpv", xtitle="track dz(PV) (cm)", ytitle="duplicates rate vs dz(PV)", title="", ymax=_maxFake),
    Plot("pileuprate_dzpv", xtitle="track dz(PV) (cm)", ytitle="pileup rate vs dz(PV)", title="", ymax=_maxFake),
], ncols=3,
                         legendDy=0.09
)
_dupandfake3 = PlotGroup("dupandfake3", [
    Plot("fakerate_vs_hit", xtitle="track hits", ytitle="fakerate vs hits", ymax=_maxFake),
    Plot("duplicatesRate_hit", xtitle="track hits", ytitle="duplicates rate vs hits", ymax=_maxFake),
    Plot("pileuprate_hit", xtitle="track hits", ytitle="pileup rate vs hits", ymax=_maxFake),
    #
    Plot("fakerate_vs_layer", xtitle="track layers", ytitle="fakerate vs layer", ymax=_maxFake, xmax=25),
    Plot("duplicatesRate_layer", xtitle="track layers", ytitle="duplicates rate vs layers", ymax=_maxFake, xmax=25),
    Plot("pileuprate_layer", xtitle="track layers", ytitle="pileup rate vs layers", ymax=_maxFake, xmax=25),
    #
    Plot("fakerate_vs_pixellayer", xtitle="track pixel layers", ytitle="fakerate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    Plot("duplicatesRate_pixellayer", xtitle="track pixel layers", ytitle="duplicates rate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    Plot("pileuprate_pixellayer", xtitle="track pixel layers", ytitle="pileup rate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    #
    Plot("fakerate_vs_3Dlayer", xtitle="track 3D layers", ytitle="fakerate vs 3D layers", ymax=_maxFake, xmax=20),
    Plot("duplicatesRate_3Dlayer", xtitle="track 3D layers", ytitle="duplicates rate vs 3D layers", ymax=_maxFake, xmax=20),
    Plot("pileuprate_3Dlayer", xtitle="track 3D layers", ytitle="pileup rate vs 3D layers", ymax=_maxFake, xmax=20)
], ncols=3,
                         legendDy=0.09
)
_dupandfake4 = PlotGroup("dupandfake4", [
    Plot("fakerate_vs_chi2", xtitle="track #chi^{2}", ytitle="fakerate vs #chi^{2}", ymax=_maxFake),
    Plot("duplicatesRate_chi2", xtitle="track #chi^{2}", ytitle="duplicates rate vs #chi^{2}", ymax=_maxFake),
    Plot("pileuprate_chi2", xtitle="track #chi^{2}", ytitle="pileup rate vs #chi^{2}", ymax=_maxFake)
],
                         legendDy=-0.025
)
_common = {
    "ytitle": "Fake+pileup rate",
    "ymax": _maxFake,
    "drawStyle": "EP",
}
_common2 = {}
_common2.update(_common)
_common2["drawStyle"] = "pcolz"
_common2["ztitleoffset"] = 1.5
_common2["xtitleoffset"] = 7
_common2["ytitleoffset"] = 10
_common2["ztitleoffset"] = 6
_pvassociation1 = PlotGroup("pvassociation1", [
    Plot(ROC("effic_vs_fakepileup_dzpvcut", "effic_vs_dzpvcut", FakeDuplicate("fakepileup_vs_dzpvcut", assoc="num_assoc(recoToSim)_dzpvcut", reco="num_reco_dzpvcut", dup="num_pileup_dzpvcut")),
             xtitle="Efficiency vs. cut on dz(PV)", **_common),
    Plot(ROC("effic_vs_fakepileup2_dzpvcut", "effic_vs_dzpvcut", FakeDuplicate("fakepileup_vs_dzpvcut", assoc="num_assoc(recoToSim)_dzpvcut", reco="num_reco_dzpvcut", dup="num_pileup_dzpvcut"), zaxis=True),
             xtitle="Efficiency", ztitle="Cut on dz(PV)", **_common2),
    #
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvcut")),
             xtitle="Efficiency vs. cut on dz(PV)/dzError", **_common),
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvcut"), zaxis=True),
             xtitle="Efficiency", ztitle="Cut on dz(PV)/dzError", **_common2),
    ##
    Plot(ROC("effic_vs_fakepileup_dzpvcut_pt",  "effic_vs_dzpvcut_pt", FakeDuplicate("fakepileup_vs_dzpvcut_pt", assoc="num_assoc(recoToSim)_dzpvcut_pt", reco="num_reco_dzpvcut_pt", dup="num_pileup_dzpvcut_pt")),
             xtitle="Efficiency (p_{T} weighted) vs. cut on dz(PV)", **_common),
    Plot(ROC("effic_vs_fakepileup_dzpvcut_pt",  "effic_vs_dzpvcut_pt", FakeDuplicate("fakepileup_vs_dzpvcut_pt", assoc="num_assoc(recoToSim)_dzpvcut_pt", reco="num_reco_dzpvcut_pt", dup="num_pileup_dzpvcut_pt"), zaxis=True),
             xtitle="Efficiency (p_{T} weighted)", ztitle="Cut on dz(PV)", **_common2),
    #
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvcut")),
             xtitle="Efficiency (p_{T} weighted) vs. cut on dz(PV)/dzError", **_common),
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvcut"), zaxis=True),
             xtitle="Efficiency (p_{T} weighted)", ztitle="Cut on dz(PV)/dzError", **_common2),
], onlyForPileup=True,
                         legendDy=0.09
)
_pvassociation2 = PlotGroup("pvassociation2", [
    Plot("effic_vs_dzpvcut", xtitle="Cut on dz(PV) (cm)", ytitle="Efficiency vs. cut on dz(PV)", ymax=_maxEff),
    Plot("effic_vs_dzpvcut2", xtitle="Cut on dz(PV) (cm)", ytitle="Efficiency (excl. trk eff)", ymax=_maxEff),
    Plot("fakerate_vs_dzpvcut", xtitle="Cut on dz(PV) (cm)", ytitle="Fake rate vs. cut on dz(PV)", ymax=_maxFake),
    Plot("pileuprate_dzpvcut", xtitle="Cut on dz(PV) (cm)", ytitle="Pileup rate vs. cut on dz(PV)", ymax=_maxFake),
    #
    Plot("effic_vs_dzpvsigcut", xtitle="Cut on dz(PV)/dzError", ytitle="Efficiency vs. cut on dz(PV)/dzError", ymax=_maxEff),
    Plot("effic_vs_dzpvsigcut2", xtitle="Cut on dz(PV)/dzError", ytitle="Efficiency (excl. trk eff)", ymax=_maxEff),
    Plot("fakerate_vs_dzpvsigcut", xtitle="Cut on dz(PV)/dzError", ytitle="Fake rate vs. cut on dz(PV)/dzError", ymax=_maxFake),
    Plot("pileuprate_dzpvsigcut", xtitle="Cut on dz(PV)/dzError", ytitle="Pileup rate vs. cut on dz(PV)/dzError", ymax=_maxFake),
], onlyForPileup=True,
                         legendDy=0.09
)
_pvassociation3 = PlotGroup("pvassociation3", [
    Plot("effic_vs_dzpvcut_pt", xtitle="Cut on dz(PV) (cm)", ytitle="Efficiency (p_{T} weighted)", ymax=_maxEff),
    Plot("effic_vs_dzpvcut2_pt", xtitle="Cut on dz(PV) (cm)", ytitle="Efficiency (p_{T} weighted, excl. trk eff)", ymax=_maxEff),
    Plot("fakerate_vs_dzpvcut_pt", xtitle="Cut on dz(PV) (cm)", ytitle="Fake rate (p_{T} weighted)", ymax=_maxFake),
    Plot("pileuprate_dzpvcut_pt", xtitle="Cut on dz(PV) (cm)", ytitle="Pileup rate (p_{T} weighted)", ymax=_maxFake),
    #
    Plot("effic_vs_dzpvsigcut_pt", xtitle="Cut on dz(PV)/dzError", ytitle="Efficiency (p_{T} weighted)", ymax=_maxEff),
    Plot("effic_vs_dzpvsigcut2_pt", xtitle="Cut on dz(PV)/dzError", ytitle="Efficiency (p_{T} weighted, excl. trk eff)", ymax=_maxEff),
    Plot("fakerate_vs_dzpvsigcut_pt", xtitle="Cut on dz(PV)/dzError", ytitle="Fake rate (p_{T} weighted)", ymax=_maxFake),
    Plot("pileuprate_dzpvsigcut_pt", xtitle="Cut on dz(PV)/dzError", ytitle="Pileup rate (p_{T} weighted)", ymax=_maxFake),
], onlyForPileup=True,
                         legendDy=0.09
)


# These don't exist in FastSim
_common = {"normalizeToUnitArea": True, "stat": True, "drawStyle": "hist"}
_dedx = PlotGroup("dedx", [
    Plot("h_dedx_estim1", xtitle="dE/dx, harm2", **_common),
    Plot("h_dedx_estim2", xtitle="dE/dx, trunc40", **_common),
    Plot("h_dedx_nom1", xtitle="dE/dx number of measurements", title="", **_common),
    Plot("h_dedx_sat1", xtitle="dE/dx number of measurements with saturation", title="", **_common),
    ],
                  legendDy=-0.025
)

_chargemisid = PlotGroup("chargemisid", [
    Plot("chargeMisIdRate", xtitle="#eta", ytitle="charge mis-id rate vs #eta", ymax=0.05),
    Plot("chargeMisIdRate_Pt", xtitle="p_{T}", ytitle="charge mis-id rate vs p_{T}", xmax=300, ymax=0.1, xlog=True),
    Plot("chargeMisIdRate_hit", xtitle="hits", ytitle="charge mis-id rate vs hits", title=""),
    Plot("chargeMisIdRate_phi", xtitle="#phi", ytitle="charge mis-id rate vs #phi", title="", ymax=0.01),
    Plot("chargeMisIdRate_dxy", xtitle="dxy", ytitle="charge mis-id rate vs dxy", ymax=0.1),
    Plot("chargeMisIdRate_dz", xtitle="dz", ytitle="charge mis-id rate vs dz", ymax=0.1)
])
_common = {"stat": True, "normalizeToUnitArea": True, "ylog": True, "ymin": 1e-6, "drawStyle": "hist"}
_hitsAndPt = PlotGroup("hitsAndPt", [
    Plot("missing_inner_layers", ymax=1, **_common),
    Plot("missing_outer_layers", ymax=1, **_common),
    Plot("hits_eta", stat=True, statx=0.38, xtitle="track #eta", ytitle="<hits> vs #eta", ymin=8, ymax=24, statyadjust=[0,0,-0.15]),
    Plot("hits", stat=True, xtitle="track hits", xmin=0, xmax=40, drawStyle="hist"),
    Plot("num_simul_pT", xtitle="TP p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common),
    Plot("num_reco_pT", xtitle="track p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common)
])
_tuning = PlotGroup("tuning", [
    Plot("chi2", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=[0.1, 0.2, 0.5, 1.0001], drawStyle="hist", xtitle="#chi^{2}", ratioUncertainty=False),
    Plot("chi2_prob", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="Prob(#chi^{2})"),
    Plot("chi2mean", stat=True, title="", xtitle="#eta", ytitle="< #chi^{2} / ndf >", ymax=2.5),
    Plot("ptres_vs_eta_Mean", stat=True, scale=100, title="", xtitle="#eta", ytitle="< #delta p_{T} / p_{T} > [%]", ymin=-1.5, ymax=1.5)
])
_common = {"stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "xmin": -10, "xmax": 10, "ylog": True, "ymin": 5e-5, "ymax": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025], "ratioUncertainty": False}
_pulls = PlotGroup("pulls", [
    Plot("pullPt", **_common),
    Plot("pullQoverp", **_common),
    Plot("pullPhi", **_common),
    Plot("pullTheta", **_common),
    Plot("pullDxy", **_common),
    Plot("pullDz", **_common),
],
                   legendDx=0.1, legendDw=-0.1, legendDh=-0.015
)
_common = {"title": "", "ylog": True, "xtitle": "#eta", "ymin": _minMaxResol, "ymax": _minMaxResol}
_resolutionsEta = PlotGroup("resolutionsEta", [
    Plot("phires_vs_eta_Sigma", ytitle="#sigma(#delta #phi) [rad]", **_common),
    Plot("cotThetares_vs_eta_Sigma", ytitle="#sigma(#delta cot(#theta))", **_common),
    Plot("dxyres_vs_eta_Sigma", ytitle="#sigma(#delta d_{0}) [cm]", **_common),
    Plot("dzres_vs_eta_Sigma", ytitle="#sigma(#delta z_{0}) [cm]", **_common),
    Plot("ptres_vs_eta_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", **_common),
])
_common = {"title": "", "ylog": True, "xlog": True, "xtitle": "p_{T}", "xmin": 0.1, "xmax": 1000, "ymin": _minMaxResol, "ymax": _minMaxResol}
_resolutionsPt = PlotGroup("resolutionsPt", [
    Plot("phires_vs_pt_Sigma", ytitle="#sigma(#delta #phi) [rad]", **_common),
    Plot("cotThetares_vs_pt_Sigma", ytitle="#sigma(#delta cot(#theta))", **_common),
    Plot("dxyres_vs_pt_Sigma", ytitle="#sigma(#delta d_{0}) [cm]", **_common),
    Plot("dzres_vs_pt_Sigma", ytitle="#sigma(#delta z_{0}) [cm]", **_common),
    Plot("ptres_vs_pt_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", **_common),
])

########################################
#
# Summary plots
#
########################################

_possibleTrackingColls = [
    'initialStep',
    'lowPtTripletStep',
    'pixelPairStep',
    'detachedTripletStep',
    'mixedTripletStep',
    'pixelLessStep',
    'tobTecStep',
    'jetCoreRegionalStep',
    'muonSeededStepInOut',
    'muonSeededStepOutIn',
    'ak4PFJets',
    'btvLike',
]
def _mapCollectionToAlgoQuality(collName):
    if "Hp" in collName:
        quality = "highPurity"
    else:
        quality = ""
    collNameLow = collName.replace("Hp", "").lower()

    algo = None
    if "general" in collNameLow or collNameLow in ["cutsreco", "cutsrecofrompv", "cutsrecofrompvalltp",
                                                   "cutsrecotracks", "custrecotracksfrompv", "cutsrecotracksfrompvalltp"]:
        algo = "ootb"
    else:
        for coll in _possibleTrackingColls:
            if coll.lower() in collNameLow:
                algo = coll
                break
        # fallback
        if algo is None:
            algo = collName

    return (algo, quality)

def _collhelper(name):
    return (name, [name])
_collLabelMap = collections.OrderedDict(map(_collhelper, ["generalTracks"]+_possibleTrackingColls))
_collLabelMapHp = collections.OrderedDict(map(_collhelper, ["generalTracks"]+filter(lambda n: "Step" in n, _possibleTrackingColls)))
def _summaryBinRename(binLabel, highPurity):
    (algo, quality) = _mapCollectionToAlgoQuality(binLabel)
    ret = None
    if highPurity:
        if quality == "highPurity":
            ret = algo
    else:
        if quality == "":
            ret = algo
    if ret == "ootb":
        ret = "generalTracks"
    return ret

_common = {"drawStyle": "EP", "xbinlabelsize": 10, "xbinlabeloption": "d"}
_commonAB = {"mapping": _collLabelMap,
             "renameBin": lambda bl: _summaryBinRename(bl, False)}
_commonN = {"ylog": True, "ymin": _minMaxN, "ymax": _minMaxN}
_commonN.update(_common)
_summary = PlotGroup("summary", [
    Plot(AggregateBins("efficiency", "effic_vs_coll", **_commonAB),
         title="Efficiency vs collection", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),
    Plot(AggregateBins("efficiencyAllPt", "effic_vs_coll_allPt", **_commonAB),
         title="Efficiency vs collection (no pT cut in denominator)", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),

    Plot(AggregateBins("fakerate", "fakerate_vs_coll", **_commonAB), title="Fakerate vs collection", ytitle="Fake rate", ymax=_maxFake, **_common),
    Plot(AggregateBins("duplicatesRate", "duplicatesRate_coll", **_commonAB), title="Duplicates rate vs collection", ytitle="Duplicates rate", ymax=_maxFake, **_common),
    Plot(AggregateBins("pileuprate", "pileuprate_coll", **_commonAB), title="Pileup rate vs collection", ytitle="Pileup rate", ymax=_maxFake, **_common),
])
_summaryN = PlotGroup("summary_ntracks", [
    Plot(AggregateBins("num_reco_coll", "num_reco_coll", **_commonAB), ytitle="Tracks", title="Number of tracks vs collection", **_commonN),
    Plot(AggregateBins("num_signal_coll", "num_assoc(recoToSim)_coll", **_commonAB), ytitle="Signal tracks", title="Number of signal tracks vs collection", **_commonN),
    Plot(AggregateBins("num_fake_coll", Subtract("num_fake_coll_orig", "num_reco_coll", "num_assoc(recoToSim)_coll"), **_commonAB), ytitle="Fake tracks", title="Number of fake tracks vs collection", **_commonN),
    Plot(AggregateBins("num_pileup_coll", "num_pileup_coll", **_commonAB), ytitle="Pileup tracks", title="Number of pileup tracks vs collection", **_commonN),
    Plot(AggregateBins("num_duplicate_coll", "num_duplicate_coll", **_commonAB), ytitle="Duplicate tracks", title="Number of duplicate tracks vs collection", **_commonN),
])
_commonAB = {"mapping": _collLabelMapHp,
             "renameBin": lambda bl: _summaryBinRename(bl, True)}
_summaryHp = PlotGroup("summary", [
    Plot(AggregateBins("efficiency", "effic_vs_coll", **_commonAB),
         title="Efficiency vs collection", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),
    Plot(AggregateBins("efficiencyefficiencyAllPt", "effic_vs_coll", **_commonAB),
         title="Efficiency vs collection (no pT cut in denominator)", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),
    Plot(AggregateBins("fakerate", "fakerate_vs_coll", **_commonAB), title="Fakerate vs collection", ytitle="Fake rate", ymax=_maxFake, **_common),
    Plot(AggregateBins("duplicatesRate", "duplicatesRate_coll", **_commonAB), title="Duplicates rate vs collection", ytitle="Duplicates rate", ymax=_maxFake, **_common),
    Plot(AggregateBins("pileuprate", "pileuprate_coll", **_commonAB), title="Pileup rate vs collection", ytitle="Pileup rate", ymax=_maxFake, **_common),
])
_summaryNHp = PlotGroup("summary_ntracks", [
    Plot(AggregateBins("num_reco_coll", "num_reco_coll", **_commonAB), ytitle="Tracks", title="Number of tracks vs collection", **_commonN),
    Plot(AggregateBins("num_signal_coll", "num_assoc(recoToSim)_coll", **_commonAB), ytitle="Signal tracks", title="Number of signal tracks vs collection", **_commonN),
    Plot(AggregateBins("num_fake_coll", Subtract("num_fake_coll_orig", "num_reco_coll", "num_assoc(recoToSim)_coll"), **_commonAB), ytitle="Fake tracks", title="Number of fake tracks vs collection", **_commonN),
    Plot(AggregateBins("num_pileup_coll", "num_pileup_coll", **_commonAB), ytitle="Pileup tracks", title="Number of pileup tracks vs collection", **_commonN),
    Plot(AggregateBins("num_duplicate_coll", "num_duplicate_coll", **_commonAB), ytitle="Duplicate tracks", title="Number of duplicate tracks vs collection", **_commonN),
])

########################################
#
# PackedCandidate plots
#
########################################

_common = {"normalizeToUnitArea": True, "ylog": True, "ymin": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "ymax": [1e-2, 1e-1, 1.1]}
_packedCandidateFlow = PlotGroup("flow", [
    Plot("selectionFlow", xbinlabelsize=10, xbinlabeloption="d", drawStyle="hist", ylog=True, ymin=[0.9, 9, 9e1, 9e2, 9e3, 9e4, 9e5, 9e6, 9e7]),
    Plot("diffCharge", xtitle="Charge", **_common),
    Plot("diffIsHighPurity", xtitle="High purity status", **_common),
    Plot("diffNdof", xtitle="ndof", **_common),
    Plot("diffNormalizedChi2", xtitle="#chi^{2}/ndof", **_common),
])
_packedCandidateHits = PlotGroup("hits", [
    Plot("diffHitPatternNumberOfValidHits", xtitle="Valid hits (via HitPattern)", **_common),
    Plot("diffHitPatternNumberOfValidPixelHits", xtitle="Valid pixel hits (via HitPattern)", **_common),
    Plot("diffHitPatternHasValidHitInFirstPixelBarrel", xtitle="Has valid hit in BPix1 layer (via HitPattern)", **_common),
    Plot("diffHitPatternNumberOfLostPixelHits", xtitle="Lost pixel hits (via HitPattern)", **_common),
    Plot("diffNumberOfHits", xtitle="Hits",  **_common),
    Plot("diffNumberOfPixelHits", xtitle="Pixel hits", **_common),
    Plot("diffLostInnerHits", xtitle="Lost inner hits", **_common),
],
                                 legendDy=0.09
)

_common["xlabelsize"] = 16
_packedCandidateMomVert = PlotGroup("momentumVertex", [
    Plot("diffPx", xtitle="p_{x}", **_common),
    Plot("diffVx", xtitle="Reference point x", **_common),
    Plot("diffPy", xtitle="p_{y}", **_common),
    Plot("diffVy", xtitle="Reference point y", **_common),
    Plot("diffPz", xtitle="p_{z}", **_common),
    Plot("diffVz", xtitle="Reference point z", **_common),
])

_common["adjustMarginRight"] = 0.05
_packedCandidateParam1 = PlotGroup("param1", [
    Plot("diffPt", xtitle="p_{T}", **_common),
    Plot("diffPtError", xtitle="p_{T} error", **_common),
    Plot("diffEta", xtitle="#eta", **_common),
    Plot("diffEtaError", xtitle="#eta error", **_common),
    Plot("diffPhi", xtitle="#phi", **_common),
    Plot("diffPhiError", xtitle="#phi error", **_common),
])
_packedCandidateParam2 = PlotGroup("param2", [
    Plot("diffDxy", xtitle="d_{xy}", **_common),
    Plot("diffDxyError", xtitle="d_{xy} error", **_common),
    Plot("diffDz", xtitle="d_{z}", **_common),
    Plot("diffDzError", xtitle="d_{z} error", **_common),
    Plot("diffQoverp", xtitle="Q/p", **_common),
    Plot("diffQoverpError", xtitle="Q/p error", **_common),
    Plot("diffTheta", xtitle="#theta", **_common),
    Plot("diffThetaError", xtitle="#theta error", **_common),
],
                                   legendDy=0.09
)

class TrackingPlotFolder(PlotFolder):
    def _init__(self, *args, **kwargs):
        super(TrackingPlotFolder, self).__init__(*args, **kwargs)

    def translateSubFolder(self, dqmSubFolderName):
        spl = dqmSubFolderName.split("_")
        if len(spl) != 2:
            return None
        collName = spl[0]
        return _mapCollectionToAlgoQuality(collName)

    def getSelectionName(self, plotFolderName, translatedDqmSubFolder):
        (algo, quality) = translatedDqmSubFolder

        ret = ""
        if plotFolderName != "":
            ret += "_"+plotFolderName
        if quality != "":
            ret += "_"+quality
        if not (algo == "ootb" and quality != ""):
            ret += "_"+algo

        return ret

    def limitSubFolder(self, limitOnlyTo, translatedDqmSubFolder):
        """Return True if this subfolder should be processed

        Arguments:
        limitOnlyTo            -- Function '(algo, quality) -> bool'
        translatedDqmSubFolder -- Return value of translateSubFolder
        """
        (algo, quality) = translatedDqmSubFolder
        return limitOnlyTo(algo, quality)

def _trackingFolders(lastDirName="Track"):
    return [
        "DQMData/Run 1/Tracking/Run summary/"+lastDirName,
        "DQMData/Tracking/"+lastDirName,
        "DQMData/Run 1/RecoTrackV/Run summary/"+lastDirName,
        "DQMData/RecoTrackV/"+lastDirName,
    ]

_simBasedPlots = [
    _effandfake1,
    _effandfake2,
    _effandfake3,
    _effvspos,
]
_recoBasedPlots = [
    _dupandfake1,
    _dupandfake2,
    _dupandfake3,
    _dupandfake4,
    _pvassociation1,
    _pvassociation2,
    _pvassociation3,
    _dedx,
#    _chargemisid,
    _hitsAndPt,
    _pulls,
    _resolutionsEta,
    _resolutionsPt,
    _tuning,
]
_summaryPlots = [
    _summary,
    _summaryN,
]
_summaryPlotsHp = [
    _summaryHp,
    _summaryNHp,
]
_packedCandidatePlots = [
    _packedCandidateFlow,
    _packedCandidateParam1,
    _packedCandidateParam2,
    _packedCandidateMomVert,
    _packedCandidateHits,
]
plotter = Plotter()
def _appendTrackingPlots(lastDirName, name, algoPlots, onlyForPileup=False):
    # to keep backward compatibility, this set of plots has empty name
    plotter.append(name, _trackingFolders(lastDirName), TrackingPlotFolder(*algoPlots, onlyForPileup=onlyForPileup, purpose=PlotPurpose.TrackingIteration))
    summaryName = ""
    if name != "":
        summaryName += name+"_"
    summaryName += "summary"
    plotter.append(summaryName, _trackingFolders(lastDirName),
                   PlotFolder(*_summaryPlots, loopSubFolders=False, onlyForPileup=onlyForPileup,
                              purpose=PlotPurpose.TrackingSummary, page="summary", section=name))
    plotter.append(summaryName+"_highPurity", _trackingFolders(lastDirName),
                   PlotFolder(*_summaryPlotsHp, loopSubFolders=False, onlyForPileup=onlyForPileup,
                              purpose=PlotPurpose.TrackingSummary, page="summary",
                              section=name+"_highPurity" if name != "" else "highPurity"),
                   fallbackNames=[summaryName]) # backward compatibility for release validation, the HP plots used to be in the same directory with all-track plots
_appendTrackingPlots("Track", "", _simBasedPlots+_recoBasedPlots)
_appendTrackingPlots("TrackAllTPEffic", "allTPEffic", _simBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPV", "fromPV", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPVAllTP", "fromPVAllTP", _recoBasedPlots, onlyForPileup=True)

# MiniAOD
plotter.append("packedCandidate", _trackingFolders("PackedCandidate"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate"))
plotter.append("packedCandidateLostTracks", _trackingFolders("PackedCandidate/lostTracks"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate (lostTracks)"))

_iterModuleMap = collections.OrderedDict([
    ("initialStepPreSplitting", ["initialStepSeedLayersPreSplitting",
                                 "initialStepSeedsPreSplitting",
                                 "initialStepTrackCandidatesPreSplitting",
                                 "initialStepTracksPreSplitting",
                                 "firstStepPrimaryVerticesPreSplitting",
                                 "iter0TrackRefsForJetsPreSplitting",
                                 "caloTowerForTrkPreSplitting",
                                 "ak4CaloJetsForTrkPreSplitting",
                                 "jetsForCoreTrackingPreSplitting",
                                 "siPixelClusters",
                                 "siPixelRecHits",
                                 "MeasurementTrackerEvent",
                                 "siPixelClusterShapeCache"]),
    ("initialStep", ['initialStepClusters',
                     'initialStepSeedLayers',
                     'initialStepSeeds',
                     'initialStepTrackCandidates',
                     'initialStepTracks',
                     'initialStepSelector',
                     'initialStep']),
    ("lowPtTripletStep", ['lowPtTripletStepClusters',
                          'lowPtTripletStepSeedLayers',
                          'lowPtTripletStepSeeds',
                          'lowPtTripletStepTrackCandidates',
                          'lowPtTripletStepTracks',
                          'lowPtTripletStepSelector']),
    ("pixelPairStep", ['pixelPairStepClusters',
                       'pixelPairStepSeedLayers',
                       'pixelPairStepSeeds',
                       'pixelPairStepTrackCandidates',
                       'pixelPairStepTracks',
                       'pixelPairStepSelector']),
    ("detachedTripletStep", ['detachedTripletStepClusters',
                             'detachedTripletStepSeedLayers',
                             'detachedTripletStepSeeds',
                             'detachedTripletStepTrackCandidates',
                             'detachedTripletStepTracks',
                             'detachedTripletStepSelector',
                             'detachedTripletStep']),
    ("mixedTripletStep", ['mixedTripletStepClusters',
                          'mixedTripletStepSeedLayersA',
                          'mixedTripletStepSeedLayersB',
                          'mixedTripletStepSeedsA',
                          'mixedTripletStepSeedsB',
                          'mixedTripletStepSeeds',
                          'mixedTripletStepTrackCandidates',
                          'mixedTripletStepTracks',
                          'mixedTripletStepSelector',
                          'mixedTripletStep']),
    ("pixelLessStep", ['pixelLessStepClusters',
                       'pixelLessStepSeedClusters',
                       'pixelLessStepSeedLayers',
                       'pixelLessStepSeeds',
                       'pixelLessStepTrackCandidates',
                       'pixelLessStepTracks',
                       'pixelLessStepSelector',
                       'pixelLessStep']),
    ("tobTecStep", ['tobTecStepClusters',
                    'tobTecStepSeedClusters',
                    'tobTecStepSeedLayersTripl',
                    'tobTecStepSeedLayersPair',
                    'tobTecStepSeedsTripl',
                    'tobTecStepSeedsPair',
                    'tobTecStepSeeds',
                    'tobTecStepTrackCandidates',
                    'tobTecStepTracks',
                    'tobTecStepSelector']),
    ("jetCoreRegionalStep", ['iter0TrackRefsForJets',
                             'caloTowerForTrk',
                             'ak4CaloJetsForTrk',
                             'jetsForCoreTracking',
                             'firstStepPrimaryVertices',
                             'firstStepGoodPrimaryVertices',
                             'jetCoreRegionalStepSeedLayers',
                             'jetCoreRegionalStepSeeds',
                             'jetCoreRegionalStepTrackCandidates',
                             'jetCoreRegionalStepTracks',
                             'jetCoreRegionalStepSelector']),
    ("muonSeededStep", ['earlyMuons',
                        'muonSeededSeedsInOut',
                        'muonSeededSeedsInOut',
                        'muonSeededTracksInOut',
                        'muonSeededSeedsOutIn',
                        'muonSeededTrackCandidatesOutIn',
                        'muonSeededTracksOutIn',
                        'muonSeededTracksInOutSelector',
                        'muonSeededTracksOutInSelector']),
])


_timing = PlotGroup("", [
    Plot(AggregateBins("iterative", "reconstruction_step_module_average", _iterModuleMap), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST", xbinlabelsize=0.03),
#    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap), ytitle="Average processing time", title="Average processing time / event (normalized)", drawStyle="HIST", xbinlabelsize=0.03, normalizeToUnitArea=True)
    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap, normalizeTo="ak7CaloJets"), ytitle="Average processing time / ak7CaloJets", title="Average processing time / event (normalized to ak7CaloJets)", drawStyle="HIST", xbinlabelsize=0.03)

    ],
                    legendDx=-0.1, legendDw=-0.35, legendDy=0.39,
)
_pixelTiming = PlotGroup("pixelTiming", [
    Plot(AggregateBins("pixel", "reconstruction_step_module_average", {"pixelTracks": ["pixelTracks"]}), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST")
])

timePlotter = Plotter()
timePlotter.append("timing", [
    "DQMData/Run 1/DQM/Run summary/TimerService/Paths",
    "DQMData/Run 1/DQM/Run summary/TimerService/process RECO/Paths",
], PlotFolder(
    _timing
#    _pixelTiming
))

_common = {"stat": True, "normalizeToUnitArea": True, "drawStyle": "hist"}
_tplifetime = PlotGroup("tplifetime", [
    Plot("TPlip", xtitle="TP lip", **_common),
    Plot("TPtip", xtitle="TP tip", **_common),
])

tpPlotter = Plotter()
tpPlotter.append("tp", [
    "DQMData/Run 1/Tracking/Run summary/TrackingMCTruth/TrackingParticle",
    "DQMData/Tracking/TrackingMCTruth/TrackingParticle",
], PlotFolder(
    _tplifetime,
))


