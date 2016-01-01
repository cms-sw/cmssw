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

_maxLayers = 25
_maxPixelLayers = 8
_max3DLayers = 20

_legendDy_4rows = 0.09
_legendDy_2rows = -0.025
_legendDy_2rows_3cols = -0.17


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
                         legendDy=_legendDy_4rows
)
_effandfake3 = PlotGroup("effandfake3", [
    Plot("effic_vs_hit", xtitle="TP hits", ytitle="efficiency vs hits", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_hit", assoc="num_assoc(recoToSim)_hit", dup="num_duplicate_hit", reco="num_reco_hit", title="fake+duplicates vs hit"),
         xtitle="track hits", ytitle="fake+duplicates rate vs hits", ymax=_maxFake),
    Plot("effic_vs_layer", xtitle="TP layers", ytitle="efficiency vs layers", xmax=_maxLayers, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_layer", assoc="num_assoc(recoToSim)_layer", dup="num_duplicate_layer", reco="num_reco_layer", title="fake+duplicates vs layer"),
         xtitle="track layers", ytitle="fake+duplicates rate vs layers", xmax=_maxLayers, ymax=_maxFake),
    Plot("effic_vs_pixellayer", xtitle="TP pixel layers", ytitle="efficiency vs pixel layers", title="", xmax=_maxPixelLayers, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_pixellayer", assoc="num_assoc(recoToSim)_pixellayer", dup="num_duplicate_pixellayer", reco="num_reco_pixellayer", title=""),
         xtitle="track pixel layers", ytitle="fake+duplicates rate vs pixel layers", xmax=_maxPixelLayers, ymax=_maxFake),
    Plot("effic_vs_3Dlayer", xtitle="TP 3D layers", ytitle="efficiency vs 3D layers", xmax=_max3DLayers, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_3Dlayer", assoc="num_assoc(recoToSim)_3Dlayer", dup="num_duplicate_3Dlayer", reco="num_reco_3Dlayer", title="fake+duplicates vs 3D layer"),
         xtitle="track 3D layers", ytitle="fake+duplicates rate vs 3D layers", xmax=_max3DLayers, ymax=_maxFake),
],
                         legendDy=_legendDy_4rows
)
_common = {"ymin": 0, "ymax": _maxEff}
_effandfake4 = PlotGroup("effandfake4", [
    Plot("effic_vs_vertpos", xtitle="TP vert xy (cm)", ytitle="efficiency vs vert xy", **_common),
    Plot(FakeDuplicate("fakeduprate_vs_vertpos", assoc="num_assoc(recoToSim)_vertpos", dup="num_duplicate_vertpos", reco="num_reco_vertpos", title="fake+duplicates vs. ref. point xy"),
         xtitle="track ref. point xy (cm)", ytitle="fake+duplicates vs xy", ymax=_maxFake),
    Plot("effic_vs_zpos", xtitle="TP vert z (cm)", ytitle="efficiency vs vert z", **_common),
    Plot(FakeDuplicate("fakeduprate_vs_zpos", assoc="num_assoc(recoToSim)_zpos", dup="num_duplicate_zpos", reco="num_reco_zpos", title="fake+duplicates vs. ref. point z"),
         xtitle="track ref. point z (cm)", ytitle="fake+duplicates vs z", ymax=_maxFake),
    Plot("effic_vs_dr", xlog=True, xtitle="TP min #DeltaR", ytitle="efficiency vs dr", **_common),
    Plot(FakeDuplicate("fakeduprate_vs_dr", assoc="num_assoc(recoToSim)_dr", dup="num_duplicate_dr", reco="num_reco_dr", title="fake+duplicates vs. #DeltaR"),
         xtitle="track min #DeltaR", ytitle="fake+duplicates vs #DeltaR", xlog=True, ymax=_maxFake),
])

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
],
                         ncols=3, legendDy=_legendDy_4rows
)
_dupandfake3 = PlotGroup("dupandfake3", [
    Plot("fakerate_vs_hit", xtitle="track hits", ytitle="fakerate vs hits", ymax=_maxFake),
    Plot("duplicatesRate_hit", xtitle="track hits", ytitle="duplicates rate vs hits", ymax=_maxFake),
    Plot("pileuprate_hit", xtitle="track hits", ytitle="pileup rate vs hits", ymax=_maxFake),
    #
    Plot("fakerate_vs_layer", xtitle="track layers", ytitle="fakerate vs layer", ymax=_maxFake, xmax=_maxLayers),
    Plot("duplicatesRate_layer", xtitle="track layers", ytitle="duplicates rate vs layers", ymax=_maxFake, xmax=_maxLayers),
    Plot("pileuprate_layer", xtitle="track layers", ytitle="pileup rate vs layers", ymax=_maxFake, xmax=_maxLayers),
    #
    Plot("fakerate_vs_pixellayer", xtitle="track pixel layers", ytitle="fakerate vs pixel layers", title="", ymax=_maxFake, xmax=_maxPixelLayers),
    Plot("duplicatesRate_pixellayer", xtitle="track pixel layers", ytitle="duplicates rate vs pixel layers", title="", ymax=_maxFake, xmax=_maxPixelLayers),
    Plot("pileuprate_pixellayer", xtitle="track pixel layers", ytitle="pileup rate vs pixel layers", title="", ymax=_maxFake, xmax=_maxPixelLayers),
    #
    Plot("fakerate_vs_3Dlayer", xtitle="track 3D layers", ytitle="fakerate vs 3D layers", ymax=_maxFake, xmax=_max3DLayers),
    Plot("duplicatesRate_3Dlayer", xtitle="track 3D layers", ytitle="duplicates rate vs 3D layers", ymax=_maxFake, xmax=_max3DLayers),
    Plot("pileuprate_3Dlayer", xtitle="track 3D layers", ytitle="pileup rate vs 3D layers", ymax=_maxFake, xmax=_max3DLayers)
],
                         ncols=3, legendDy=_legendDy_4rows
)
_dupandfake4 = PlotGroup("dupandfake4", [
    Plot("fakerate_vs_vertpos", xtitle="track ref. point xy (cm)", ytitle="fakerate vs xy", ymax=_maxFake),
    Plot("duplicatesRate_vertpos", xtitle="track ref. point xy (cm)", ytitle="duplicates rate vs xy", ymax=_maxFake),
    Plot("pileuprate_vertpos", xtitle="track ref. point xy (cm)", ytitle="pileup rate vs xy", ymax=_maxFake),
    #
    Plot("fakerate_vs_zpos", xtitle="track ref. point z (cm)", ytitle="fakerate vs z", ymax=_maxFake),
    Plot("duplicatesRate_zpos", xtitle="track ref. point z (cm)", ytitle="duplicates rate vs z", ymax=_maxFake),
    Plot("pileuprate_zpos", xtitle="track ref. point z (cm)", ytitle="pileup rate vs z", ymax=_maxFake),
    #
    Plot("fakerate_vs_dr", xtitle="track min #DeltaR", ytitle="fakerate vs #DeltaR", xlog=True, ymax=_maxFake),
    Plot("duplicatesRate_dr", xtitle="track min #DeltaR", ytitle="duplicates rate vs #DeltaR", xlog=True, ymax=_maxFake),
    Plot("pileuprate_dr", xtitle="track min #DeltaR", ytitle="pileup rate vs #DeltaR", xlog=True, ymax=_maxFake),
    #
    Plot("fakerate_vs_chi2", xtitle="track #chi^{2}", ytitle="fakerate vs #chi^{2}", ymax=_maxFake),
    Plot("duplicatesRate_chi2", xtitle="track #chi^{2}", ytitle="duplicates rate vs #chi^{2}", ymax=_maxFake),
    Plot("pileuprate_chi2", xtitle="track #chi^{2}", ytitle="pileup rate vs #chi^{2}", ymax=_maxFake)
],
                         ncols=3, legendDy=_legendDy_4rows
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
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvsigcut")),
             xtitle="Efficiency vs. cut on dz(PV)/dzError", **_common),
    Plot(ROC("effic_vs_fakepileup2_dzpvsigcut",  "effic_vs_dzpvsigcut", FakeDuplicate("fakepileup_vs_dzpvsigcut", assoc="num_assoc(recoToSim)_dzpvsigcut", reco="num_reco_dzpvsigcut", dup="num_pileup_dzpvsigcut"), zaxis=True),
             xtitle="Efficiency", ztitle="Cut on dz(PV)/dzError", **_common2),
    ##
    Plot(ROC("effic_vs_fakepileup_dzpvcut_pt",  "effic_vs_dzpvcut_pt", FakeDuplicate("fakepileup_vs_dzpvcut_pt", assoc="num_assoc(recoToSim)_dzpvcut_pt", reco="num_reco_dzpvcut_pt", dup="num_pileup_dzpvcut_pt")),
             xtitle="Efficiency (p_{T} weighted) vs. cut on dz(PV)", **_common),
    Plot(ROC("effic_vs_fakepileup2_dzpvcut_pt",  "effic_vs_dzpvcut_pt", FakeDuplicate("fakepileup_vs_dzpvcut_pt", assoc="num_assoc(recoToSim)_dzpvcut_pt", reco="num_reco_dzpvcut_pt", dup="num_pileup_dzpvcut_pt"), zaxis=True),
             xtitle="Efficiency (p_{T} weighted)", ztitle="Cut on dz(PV)", **_common2),
    #
    Plot(ROC("effic_vs_fakepileup_dzpvsigcut_pt",  "effic_vs_dzpvsigcut_pt", FakeDuplicate("fakepileup_vs_dzpvsigcut_pt", assoc="num_assoc(recoToSim)_dzpvsigcut_pt", reco="num_reco_dzpvsigcut_pt", dup="num_pileup_dzpvsigcut_pt")),
             xtitle="Efficiency (p_{T} weighted) vs. cut on dz(PV)/dzError", **_common),
    Plot(ROC("effic_vs_fakepileup2_dzpvsigcut_pt",  "effic_vs_dzpvsigcut_pt", FakeDuplicate("fakepileup_vs_dzpvsigcut_pt", assoc="num_assoc(recoToSim)_dzpvsigcut_pt", reco="num_reco_dzpvsigcut_pt", dup="num_pileup_dzpvsigcut_pt"), zaxis=True),
             xtitle="Efficiency (p_{T} weighted)", ztitle="Cut on dz(PV)/dzError", **_common2),
], onlyForPileup=True,
                         legendDy=_legendDy_4rows
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
                         legendDy=_legendDy_4rows
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
                         legendDy=_legendDy_4rows
)


# These don't exist in FastSim
_common = {"normalizeToUnitArea": True, "stat": True, "drawStyle": "hist"}
_dedx = PlotGroup("dedx", [
    Plot("h_dedx_estim1", xtitle="dE/dx, harm2", **_common),
    Plot("h_dedx_estim2", xtitle="dE/dx, trunc40", **_common),
    Plot("h_dedx_nom1", xtitle="dE/dx number of measurements", title="", **_common),
    Plot("h_dedx_sat1", xtitle="dE/dx number of measurements with saturation", title="", **_common),
    ],
                  legendDy=_legendDy_2rows
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
    Plot("hits_eta", stat=True, statx=0.38, xtitle="track #eta", ytitle="<hits> vs #eta", ymin=8, ymax=24, statyadjust=[0,0,-0.15],
         fallback={"name": "nhits_vs_eta", "profileX": True}),
    Plot("hits", stat=True, xtitle="track hits", xmin=0, ylog=True, ymin=[5e-1, 5, 5e1, 5e2, 5e3], drawStyle="hist"),
    Plot("num_simul_pT", xtitle="TP p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common),
    Plot("num_reco_pT", xtitle="track p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common)
])
_tuning = PlotGroup("tuning", [
    Plot("chi2", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=[0.1, 0.2, 0.5, 1.0001], drawStyle="hist", xtitle="#chi^{2}", ratioUncertainty=False),
    Plot("chi2_prob", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="Prob(#chi^{2})"),
    Plot("chi2mean", stat=True, title="", xtitle="#eta", ytitle="< #chi^{2} / ndf >", ymax=2.5,
         fallback={"name": "chi2_vs_eta", "profileX": True}),
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

_possibleTrackingIterations = [
    'initialStepPreSplitting',
    'initialStep',
    'highPtTripletStep', # phase1
    'lowPtQuadStep', # phase1
    'lowPtTripletStep',
    'pixelPairStep',
    'detachedQuadStep', # phase1
    'detachedTripletStep',
    'mixedTripletStepA', # seeds
    'mixedTripletStepB', # seeds
    'mixedTripletStep',
    'pixelLessStep',
    'tobTecStepPair',  # seeds
    'tobTecStepTripl', # seeds
    'tobTecStep',
    'jetCoreRegionalStep',
    'muonSeededStepInOut',
    'muonSeededStepOutIn',
    'duplicateMerge',
]
_possibleTrackingColls = _possibleTrackingIterations+[
    'ak4PFJets',
    'btvLike',
]
_possibleTrackingCollsOld = {
    "Zero"  : "iter0",
    "First" : "iter1",
    "Second": "iter2",
    "Third" : "iter3",
    "Fourth": "iter4",
    "Fifth" : "iter5",
    "Sixth" : "iter6",
    "Ninth" : "iter9",
    "Tenth" : "iter10",
}

def _trackingSubFoldersFallbackSLHC(subfolder):
    ret = subfolder.replace("trackingParticleRecoAsssociation", "AssociatorByHitsRecoDenom")
    for (old, new) in [("InitialStep",         "Zero"),
                       ("LowPtTripletStep",    "First"),
                       ("PixelPairStep",       "Second"),
                       ("MixedTripletStep",    "Fourth"),
                       ("MuonSeededStepInOut", "Ninth"),
                       ("MuonSeededStepOutIn", "Tenth")]:
        ret = ret.replace(old, new)
    if ret == subfolder:
        return None
    return ret
def _trackingRefFileFallbackSLHC(path):
    for (old, new) in [("initialStep",         "iter0"),
                       ("lowPtTripletStep",    "iter1"),
                       ("pixelPairStep",       "iter2"),
                       ("mixedTripletStep",    "iter4"),
                       ("muonSeededStepInOut", "iter9"),
                       ("muonSeededStepOutIn", "iter10")]:
        path = path.replace(old, new)
    return path

def _mapCollectionToAlgoQuality(collName):
    if "Hp" in collName:
        quality = "highPurity"
    else:
        quality = ""
    hasPtCut = False
    if "Pt" in collName:
        if "Step" in collName:
            hasPtCut = collName.index("Pt") > collName.index("Step")
        else:
            hasPtCut = True
    collNameNoQuality = collName.replace("Hp", "")
    if hasPtCut:
        quality += "Pt"
        collNameNoQuality = collNameNoQuality.replace("Pt", "")
    if "ByOriginalAlgo" in collName:
        quality += "ByOriginalAlgo"
        collNameNoQuality = collNameNoQuality.replace("ByOriginalAlgo", "")
    if "ByAlgoMask" in collName:
        quality += "ByAlgoMask"
        collNameNoQuality = collNameNoQuality.replace("ByAlgoMask", "")
    collNameNoQuality = collNameNoQuality.replace("Tracks", "", 1) # make summary naming consistent with iteration folders
    collNameLow = collNameNoQuality.lower().replace("frompv2", "").replace("frompv", "").replace("frompvalltp", "").replace("alltp", "")

    if collNameLow.find("seed") == 0:
        if quality != "":
            raise Exception("Assumption of empty quality for seeds failed, got quality '%s'" % quality)
        collNameLow = collNameLow[4:]
        if collNameLow == "initialstepseedspresplitting":
            collNameLow = "initialsteppresplittingseeds"
        elif collNameLow == "muonseededseedsinout":
            collNameLow = "muonseededstepinoutseeds"
        elif collNameLow == "muonseededseedsoutin":
            collNameLow = "muonseededstepoutinseeds"

        i_seeds = collNameLow.index("seeds")
        quality = collNameLow[i_seeds:]

        collNameLow = collNameLow[:i_seeds]

    algo = None
    prefixes = ["cutsreco", "cutsrecofrompv", "cutsrecofrompv2", "cutsrecofrompvalltp"]
    if collNameLow in ["general", "generalfrompv"]+prefixes:
        algo = "ootb"
    else:
        def testColl(coll):
            for pfx in prefixes:
                if coll == collNameLow.replace(pfx, ""):
                    return True
            return False

        for coll in _possibleTrackingColls:
            if testColl(coll.lower()):
                algo = coll
                break
        # next try "old style"
        if algo is None:
            for coll, name in _possibleTrackingCollsOld.iteritems():
                if testColl(coll.lower()):
                    algo = name
                    break

        # fallback
        if algo is None:
            algo = collNameNoQuality

    # fix for track collection naming convention
    if algo == "muonSeededInOut":
        algo = "muonSeededStepInOut"
    if algo == "muonSeededOutIn":
        algo = "muonSeededStepOutIn"

    return (algo, quality)

def _collhelper(name):
    return (name, [name])
_collLabelMap = collections.OrderedDict(map(_collhelper, ["generalTracks"]+_possibleTrackingColls))
_collLabelMapHp = collections.OrderedDict(map(_collhelper, ["generalTracks"]+filter(lambda n: "Step" in n, _possibleTrackingColls)))
def _summaryBinRename(binLabel, highPurity, byOriginalAlgo, byAlgoMask, seeds):
    (algo, quality) = _mapCollectionToAlgoQuality(binLabel)
    if algo == "ootb":
        algo = "generalTracks"
    ret = None

    if byOriginalAlgo:
        if algo != "generalTracks" and "ByOriginalAlgo" not in quality:
            return None
        quality = quality.replace("ByOriginalAlgo", "")
    if byAlgoMask:
        if algo != "generalTracks" and "ByAlgoMask" not in quality:
            return None
        quality = quality.replace("ByAlgoMask", "")

    if highPurity:
        if quality == "highPurity":
            ret = algo
    elif seeds:
        i_seeds = quality.find("seeds")
        if i_seeds == 0:
            ret = algo
            seedSubColl = quality[i_seeds+5:]
            if seedSubColl != "":
                ret += seedSubColl[0].upper() + seedSubColl[1:]
    else:
        if quality == "":
            ret = algo

    return ret

def _constructSummary(mapping, highPurity=False, byOriginalAlgo=False, byAlgoMask=False, seeds=False, midfix=""):
    _common = {"drawStyle": "EP", "xbinlabelsize": 10, "xbinlabeloption": "d"}
    _commonN = {"ylog": True, "ymin": _minMaxN, "ymax": _minMaxN}
    _commonN.update(_common)
    _commonAB = {"mapping": mapping,
                 "renameBin": lambda bl: _summaryBinRename(bl, highPurity, byOriginalAlgo, byAlgoMask, seeds),
                 "ignoreMissingBins": True,
    }
    if byOriginalAlgo or byAlgoMask:
        _commonAB["minExistingBins"] = 2
    prefix = "summary"+midfix
    summary = PlotGroup(prefix, [
        Plot(AggregateBins("efficiency", "effic_vs_coll", **_commonAB),
             title="Efficiency vs collection", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),
        Plot(AggregateBins("efficiencyAllPt", "effic_vs_coll_allPt", **_commonAB),
             title="Efficiency vs collection (no pT cut in denominator)", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),

        Plot(AggregateBins("fakerate", "fakerate_vs_coll", **_commonAB), title="Fakerate vs collection", ytitle="Fake rate", ymax=_maxFake, **_common),
        Plot(AggregateBins("duplicatesRate", "duplicatesRate_coll", **_commonAB), title="Duplicates rate vs collection", ytitle="Duplicates rate", ymax=_maxFake, **_common),
        Plot(AggregateBins("pileuprate", "pileuprate_coll", **_commonAB), title="Pileup rate vs collection", ytitle="Pileup rate", ymax=_maxFake, **_common),
    ])
    summaryN = PlotGroup(prefix+"_ntracks", [
        Plot(AggregateBins("num_reco_coll", "num_reco_coll", **_commonAB), ytitle="Tracks", title="Number of tracks vs collection", **_commonN),
        Plot(AggregateBins("num_true_coll", "num_assoc(recoToSim)_coll", **_commonAB), ytitle="True tracks", title="Number of true tracks vs collection", **_commonN),
        Plot(AggregateBins("num_fake_coll", Subtract("num_fake_coll_orig", "num_reco_coll", "num_assoc(recoToSim)_coll"), **_commonAB), ytitle="Fake tracks", title="Number of fake tracks vs collection", **_commonN),
        Plot(AggregateBins("num_duplicate_coll", "num_duplicate_coll", **_commonAB), ytitle="Duplicate tracks", title="Number of duplicate tracks vs collection", **_commonN),
        Plot(AggregateBins("num_pileup_coll", "num_pileup_coll", **_commonAB), ytitle="Pileup tracks", title="Number of pileup tracks vs collection", **_commonN),
    ])

    return (summary, summaryN)

(_summary,                 _summaryN)                 = _constructSummary(_collLabelMap)
(_summaryHp,               _summaryNHp)               = _constructSummary(_collLabelMapHp, highPurity=True)
(_summaryByOriginalAlgo,   _summaryByOriginalAlgoN)   = _constructSummary(_collLabelMapHp, byOriginalAlgo=True, midfix="ByOriginalAlgo")
(_summaryByOriginalAlgoHp, _summaryByOriginalAlgoNHp) = _constructSummary(_collLabelMapHp, byOriginalAlgo=True, midfix="ByOriginalAlgo", highPurity=True)
(_summaryByAlgoMask,       _summaryByAlgoMaskN)       = _constructSummary(_collLabelMapHp, byAlgoMask=True, midfix="ByAlgoMask")
(_summaryByAlgoMaskHp,     _summaryByAlgoMaskNHp)     = _constructSummary(_collLabelMapHp, byAlgoMask=True, midfix="ByAlgoMask", highPurity=True)
(_summarySeeds,            _summarySeedsN)            = _constructSummary(_collLabelMapHp, seeds=True)

########################################
#
# PackedCandidate plots
#
########################################

_common = {"normalizeToUnitArea": True, "ylog": True, "ymin": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "ymax": [1e-2, 1e-1, 1.1]}
_commonStatus = {}
_commonStatus.update(_common)
_commonStatus.update({"xbinlabelsize": 10, "drawStyle": "hist", "adjustMarginRight": 0.08})

_packedCandidateFlow = PlotGroup("flow", [
    Plot("selectionFlow", xbinlabelsize=10, xbinlabeloption="d", adjustMarginRight=0.1, drawStyle="hist", ylog=True, ymin=[0.9, 9, 9e1, 9e2, 9e3, 9e4, 9e5, 9e6, 9e7]),
    Plot("diffCharge", xtitle="Charge", **_common),
    Plot("diffIsHighPurity", xtitle="High purity status", **_common),
    Plot("diffNdof", xtitle="ndof", **_common),
    Plot("diffNormalizedChi2", xtitle="#chi^{2}/ndof", **_common),
])

_packedCandidateHitsHitPattern = PlotGroup("hitsHitPattern", [
    Plot("diffHitPatternNumberOfValidHits", xtitle="Valid hits (via HitPattern)", **_common),
    Plot("diffHitPatternNumberOfValidPixelHits", xtitle="Valid pixel hits (via HitPattern)", **_common),
    Plot("diffHitPatternHasValidHitInFirstPixelBarrel", xtitle="Has valid hit in BPix1 layer (via HitPattern)", **_common),
    Plot("diffHitPatternNumberOfLostPixelHits", xtitle="Lost pixel hits (via HitPattern)", **_common),
],
                                           legendDy=_legendDy_2rows
)
_packedCandidateHits = PlotGroup("hits", [
    Plot("diffNumberOfHits", xtitle="Hits",  **_common),
    Plot("diffNumberOfPixelHits", xtitle="Pixel hits", **_common),
    Plot("diffLostInnerHits", xtitle="Lost inner hits", **_common),
    Plot("numberHitsOverMax", xtitle="Number of overflown hits", **_common),
    Plot("numberPixelHitsOverMax", xtitle="Number of overflown pixel hits", **_common),
    Plot("numberStripHitsOverMax", xtitle="Number of overflown strip hits", **_common),
],
                                 ncols=3, legendDy=_legendDy_2rows_3cols
)

_packedCandidateImpactParameter1 = PlotGroup("impactParameter1", [
    Plot("diffDxyAssocPV", xtitle="dxy(assocPV)", **_common),
    Plot("diffDxyAssocPVStatus", **_commonStatus),
    Plot("diffDxyAssocPVUnderOverFlowSign", xtitle="dxy(assocPV)", **_common),
    Plot("diffDzAssocPV", xtitle="dz(assocPV)", **_common),
    Plot("diffDzAssocPVStatus", **_commonStatus),
    Plot("diffDzAssocPVUnderOverFlowSign", xtitle="dz(assocPV)", **_common),
    Plot("diffDxyError", xtitle="dxyError()", **_common),
    Plot("diffDszError", xtitle="dszError()", **_common),
    Plot("diffDzError", xtitle="dzError()", **_common),

],
                                             ncols=3
)

_packedCandidateImpactParameter2 = PlotGroup("impactParameter2", [
    Plot("diffDxyPV", xtitle="dxy(PV) via PC", **_common),
    Plot("diffDzPV", xtitle="dz(PV) via PC", **_common),
    Plot("diffTrackDxyAssocPV", xtitle="dxy(PV) via PC::bestTrack()", **_common),
    Plot("diffTrackDzAssocPV", xtitle="dz(PV) via PC::bestTrack()", **_common),
    Plot("diffTrackDxyError", xtitle="dxyError() via PC::bestTrack()", **_common),
    Plot("diffTrackDzError", xtitle="dzError() via PC::besTTrack()", **_common),
])

_packedCandidateCovarianceMatrix1 = PlotGroup("covarianceMatrix1", [
    Plot("diffCovQoverpQoverp", xtitle="cov(qoverp, qoverp)", **_common),
    Plot("diffCovQoverpQoverpStatus", **_commonStatus),
    Plot("diffCovQoverpQoverpUnderOverFlowSign", xtitle="cov(qoverp, qoverp)", **_common),
    Plot("diffCovLambdaLambda", xtitle="cov(lambda, lambda)", **_common),
    Plot("diffCovLambdaLambdaStatus", **_commonStatus),
    Plot("diffCovLambdaLambdaUnderOverFlowSign", xtitle="cov(lambda, lambda)", **_common),
    Plot("diffCovLambdaDsz", xtitle="cov(lambda, dsz)", **_common),
    Plot("diffCovLambdaDszStatus", **_commonStatus),
    Plot("diffCovLambdaDszUnderOverFlowSign", xtitle="cov(lambda, dsz)", **_common),
    Plot("diffCovPhiPhi", xtitle="cov(phi, phi)", **_common),
    Plot("diffCovPhiPhiStatus", **_commonStatus),
    Plot("diffCovPhiPhiUnderOverFlowSign", xtitle="cov(phi, phi)", **_common),
],
                                              ncols=3, legendDy=_legendDy_4rows
)
_packedCandidateCovarianceMatrix2 = PlotGroup("covarianceMatrix2", [
    Plot("diffCovPhiDxy", xtitle="cov(phi, dxy)", **_common),
    Plot("diffCovPhiDxyStatus", **_commonStatus),
    Plot("diffCovPhiDxyUnderOverFlowSign", xtitle="cov(phi, dxy)", **_common),
    Plot("diffCovDxyDxy", xtitle="cov(dxy, dxy)", **_common),
    Plot("diffCovDxyDxyStatus", **_commonStatus),
    Plot("diffCovDxyDxyUnderOverFlowSign", xtitle="cov(dxy, dxy)", **_common),
    Plot("diffCovDxyDsz", xtitle="cov(dxy, dsz)", **_common),
    Plot("diffCovDxyDszStatus", **_commonStatus),
    Plot("diffCovDxyDszUnderOverFlowSign", xtitle="cov(dxy, dsz)", **_common),
    Plot("diffCovDszDsz", xtitle="cov(dsz, dsz)", **_common),
    Plot("diffCovDszDszStatus", **_commonStatus),
    Plot("diffCovDszDszUnderOverFlowSign", xtitle="cov(dsz, dsz)", **_common),
],
                                              ncols=3, legendDy=_legendDy_4rows
)

_common["xlabelsize"] = 16
_packedCandidateVertex = PlotGroup("vertex", [
    Plot("diffVx", xtitle="Reference point x", **_common),
    Plot("diffVy", xtitle="Reference point y", **_common),
    Plot("diffVz", xtitle="Reference point z", **_common),
],
                                   legendDy=_legendDy_2rows
)

_common["adjustMarginRight"] = 0.05
_packedCandidateKinematics = PlotGroup("kinematics", [
    Plot("diffPt", xtitle="p_{T}", **_common),
    Plot("diffPtError", xtitle="p_{T} error", **_common),
    Plot("diffEta", xtitle="#eta", **_common),
    Plot("diffEtaError", xtitle="#eta error", **_common),
    Plot("diffPhi", xtitle="#phi", **_common),
    Plot("diffPhiError", xtitle="#phi error", **_common), # currently missing?
])

class TrackingPlotFolder(PlotFolder):
    def __init__(self, *args, **kwargs):
        self._fallbackRefFiles = kwargs.pop("fallbackRefFiles", [])
        PlotFolder.__init__(self, *args, **kwargs)

    def translateSubFolder(self, dqmSubFolderName):
        spl = dqmSubFolderName.split("_")
        if len(spl) != 2:
            return None
        collName = spl[0]
        return _mapCollectionToAlgoQuality(collName)

    def iterSelectionName(self, plotFolderName, translatedDqmSubFolder):
        (algoOrig, quality) = translatedDqmSubFolder

        for fallback in [lambda n: n]+self._fallbackRefFiles:
            algo = fallback(algoOrig)

            ret = ""
            if plotFolderName != "":
                ret += "_"+plotFolderName
            if quality != "":
                ret += "_"+quality
            if not (algo == "ootb" and quality != ""):
                ret += "_"+algo
            yield ret

    def limitSubFolder(self, limitOnlyTo, translatedDqmSubFolder):
        """Return True if this subfolder should be processed

        Arguments:
        limitOnlyTo            -- Function '(algo, quality) -> bool'
        translatedDqmSubFolder -- Return value of translateSubFolder
        """
        (algo, quality) = translatedDqmSubFolder
        return limitOnlyTo(algo, quality)

    # track-specific hack
    def isAlgoIterative(self, algo):
        return algo in _possibleTrackingIterations or algo in _possibleTrackingCollsOld.values()

class TrackingSummaryTable:
    def __init__(self, section, highPurity=False):
        self._highPurity = highPurity
        self._purpose = PlotPurpose.TrackingSummary
        self._page = "summary"
        self._section = section

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self, dqmSubFolder):
        return self._section

    def create(self, tdirectory):
        def _getN(hname):
            h = tdirectory.Get(hname)
            if not h:
                return None
            if self._highPurity:
                (algo, quality) = _mapCollectionToAlgoQuality(h.GetXaxis().GetBinLabel(2))
                if algo != "ootb" and quality != "highPurity":
                    return None
                return h.GetBinContent(2)
            else:
                (algo, quality) = _mapCollectionToAlgoQuality(h.GetXaxis().GetBinLabel(1))
                if algo != "ootb" and quality != "":
                    return None
                return h.GetBinContent(1)
        def _formatOrNone(num, func):
            if num is None:
                return None
            return func(num)

        n_tracks = _formatOrNone(_getN("num_reco_coll"), int)
        n_true = _formatOrNone(_getN("num_assoc(recoToSim)_coll"), int)
        if n_tracks is not None and n_true is not None:
            n_fake = n_tracks-n_true
        else:
            n_fake = None
        n_pileup = _formatOrNone(_getN("num_pileup_coll"), int)
        n_duplicate = _formatOrNone(_getN("num_duplicate_coll"), int)

        eff = _formatOrNone(_getN("effic_vs_coll"), lambda n: "%.4f" % n)

        ret = [eff, n_tracks, n_true, n_fake, n_pileup, n_duplicate]
        if ret.count(None) == len(ret):
            return None
        return ret

    def headers(self):
        return [
            "Efficiency",
            "Number of tracks",
            "Number of true tracks",
            "Number of fake tracks",
            "Number of pileup tracks",
            "Number of duplicate tracks"
        ]

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
    _effandfake4,
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
_seedingBuildingPlots = _simBasedPlots + [
    _dupandfake1,
    _dupandfake2,
    _dupandfake3,
    _dupandfake4,
    _hitsAndPt,
]
_summaryPlots = [
    _summary,
    _summaryN,
    _summaryByOriginalAlgo,
    _summaryByOriginalAlgoN,
    _summaryByAlgoMask,
    _summaryByAlgoMaskN,
]
_summaryPlotsHp = [
    _summaryHp,
    _summaryNHp,
    _summaryByOriginalAlgoHp,
    _summaryByOriginalAlgoNHp,
    _summaryByAlgoMaskHp,
    _summaryByAlgoMaskNHp,
]
_summaryPlotsSeeds = [
    _summarySeeds,
    _summarySeedsN,
]
_packedCandidatePlots = [
    _packedCandidateFlow,
    _packedCandidateKinematics,
    _packedCandidateVertex,
    _packedCandidateImpactParameter1,
    _packedCandidateImpactParameter2,
    _packedCandidateCovarianceMatrix1,
    _packedCandidateCovarianceMatrix2,
    _packedCandidateHits,
    _packedCandidateHitsHitPattern,
]
plotter = Plotter()
def _appendTrackingPlots(lastDirName, name, algoPlots, onlyForPileup=False, seeding=False):
    # to keep backward compatibility, this set of plots has empty name
    plotter.append(name, _trackingFolders(lastDirName), TrackingPlotFolder(*algoPlots, onlyForPileup=onlyForPileup, purpose=PlotPurpose.TrackingIteration, fallbackRefFiles=[_trackingRefFileFallbackSLHC]), fallbackDqmSubFolders=[_trackingSubFoldersFallbackSLHC])
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
    if seeding:
        plotter.append(summaryName+"_seeds", _trackingFolders(lastDirName),
                       PlotFolder(*_summaryPlotsSeeds, loopSubFolders=False, onlyForPileup=onlyForPileup,
                                  purpose=PlotPurpose.TrackingSummary, page="summary",
                                  section=name+"_seeds"))

    plotter.appendTable(summaryName, _trackingFolders(lastDirName), TrackingSummaryTable(section=name))
    plotter.appendTable(summaryName+"_highPurity", _trackingFolders(lastDirName), TrackingSummaryTable(section=name+"_highPurity" if name != "" else "highPurity", highPurity=True))
_appendTrackingPlots("Track", "", _simBasedPlots+_recoBasedPlots)
_appendTrackingPlots("TrackAllTPEffic", "allTPEffic", _simBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPV", "fromPV", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPVAllTP", "fromPVAllTP", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPVAllTP2", "fromPVAllTP2", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackSeeding", "seeding", _seedingBuildingPlots, seeding=True)
_appendTrackingPlots("TrackBuilding", "building", _seedingBuildingPlots)

# MiniAOD
plotter.append("packedCandidate", _trackingFolders("PackedCandidate"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate"))
plotter.append("packedCandidateLostTracks", _trackingFolders("PackedCandidate/lostTracks"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate (lostTracks)"))

# Timing
class Iteration:
    def __init__(self, name, clusterMasking=None, seeding=None, building=None, fit=None, selection=None, other=[]):
        self._name = name

        def _set(param, name, modules):
            if param is not None:
                setattr(self, name, param)
            else:
                setattr(self, name, modules)

        _set(clusterMasking, "_clusterMasking", [self._name+"Clusters"])
        _set(seeding, "_seeding", [self._name+"SeedingLayers", self._name+"Seeds"])
        _set(building, "_building", [self._name+"TrackCandidates"])
        _set(fit, "_fit", [self._name+"Tracks"])
        _set(selection, "_selection", [self._name])
        self._other = other

    def name(self):
        return self._name

    def all(self):
        return self._clusterMasking+self._seeding+self._building+self._fit+self._selection+self._other

    def clusterMasking(self):
        return self._clusterMasking

    def seeding(self):
        return self._seeding

    def building(self):
        return self._building

    def fit(self):
        return self._fit

    def selection(self):
        return self._selection

    def other(self):
        return self._other

    def modules(self):
        return [("ClusterMask", self.clusterMasking()),
                ("Seeding", self.seeding()),
                ("Building", self.building()),
                ("Fit", self.fit()),
                ("Selection", self.selection()),
                ("Other", self.other())]


_iterations = [
    Iteration("initialStepPreSplitting", clusterMasking=[],
              seeding=["initialStepSeedLayersPreSplitting",
                       "initialStepSeedsPreSplitting"],
              building=["initialStepTrackCandidatesPreSplitting"],
              fit=["initialStepTracksPreSplitting"],
              other=["firstStepPrimaryVerticesPreSplitting",
                     "initialStepTrackRefsForJetsPreSplitting",
                     "caloTowerForTrkPreSplitting",
                     "ak4CaloJetsForTrkPreSplitting",
                     "jetsForCoreTrackingPreSplitting",
                     "siPixelClusters",
                     "siPixelRecHits",
                     "MeasurementTrackerEvent",
                     "siPixelClusterShapeCache"]),
    Iteration("initialStep", clusterMasking=[],
              selection=["initialStepClassifier1",
                         "initialStepClassifier2",
                         "initialStepClassifier3",
                         "initialStep"],
              other=["firstStepPrimaryVertices"]),
    Iteration("highPtTripletStep",
              selection=["highPtTripletStepClassifier1",
                         "highPtTripletStepClassifier2",
                         "highPtTripletStepClassifier3",
                         "highPtTripletStep"]),
    Iteration("detachedQuadStep",
              selection=["detachedQuadStepClassifier1",
                         "detachedQuadStepClassifier2",
                         "detachedQuadStep"]),
    Iteration("detachedTripletStep",
              selection=["detachedTripletStepClassifier1",
                         "detachedTripletStepClassifier2",
                         "detachedTripletStep"]),
    Iteration("lowPtQuadStep"),
    Iteration("lowPtTripletStep"),
    Iteration("pixelPairStep"),
    Iteration("mixedTripletStep",
              seeding=["mixedTripletStepSeedLayersA",
                       "mixedTripletStepSeedLayersB",
                       "mixedTripletStepSeedsA",
                       "mixedTripletStepSeedsB",
                       "mixedTripletStepSeeds"],
              selection=["mixedTripletStepClassifier1",
                         "mixedTripletStepClassifier2",
                         "mixedTripletStep"]),
    Iteration("pixelLessStep",
              selection=["pixelLessStepClassifier1",
                         "pixelLessStepClassifier2",
                         "pixelLessStep"]),
    Iteration("tobTecStep",
              seeding=["tobTecStepSeedLayersTripl",
                       "tobTecStepSeedLayersPair",
                       "tobTecStepSeedsTripl",
                       "tobTecStepSeedsPair",
                       "tobTecStepSeeds"],
              selection=["tobTecStepClassifier1",
                         "tobTecStepClassifier2",
                         "tobTecStep"]),
    Iteration("jetCoreRegionalStep",
              clusterMasking=[],
              other=["initialStepTrackRefsForJets",
                     "caloJetsForTrk",
                     "jetsForCoreTracking",
                     "firstStepGoodPrimaryVertices",
                     ]),
    Iteration("muonSeededSteps",
              clusterMasking=[],
              seeding=["muonSeededSeedsInOut",
                       "muonSeededSeedsOutIn"],
              building=["muonSeededTrackCandidatesInOut",
                        "muonSeededTrackCandidatesOutIn"],
              fit=["muonSeededTracksInOut",
                   "muonSeededTracksOutIn"],
              selection=["muonSeededTracksInOutClassifier",
                         "muonSeededTracksOutIntClassifier"],
#              other=["earlyMuons"]
          ),
    Iteration("duplicateMerge",
              clusterMasking=[], seeding=[],
              building=["duplicateTrackCandidates"],
              fit=["mergedDuplicateTracks"],
              selection=["duplicateTrackClassifier"]),
    Iteration("generalTracks",
              clusterMasking=[], seeding=[], building=[], fit=[], selection=[],
              other=["preDuplicateMergingGeneralTracks",
                     "generalTracks"]),
    Iteration("ConvStep",
              clusterMasking=["convClusters"],
              seeding=["convLayerPairs",
                       "photonConvTrajSeedFromSingleLeg"],
              building=["convTrackCandidates"],
              fit=["convStepTracks"],
              selection=["convStepSelector"]),
]

def _iterModuleMap(includeConvStep=True, onlyConvStep=False):
    iterations = _iterations
    if not includeConvStep:
        iterations = filter(lambda i: i.name() != "ConvStep", iterations)
    if onlyConvStep:
        iterations = filter(lambda i: i.name() == "ConvStep", iterations)
    return collections.OrderedDict([(i.name(), i.all()) for i in iterations])
def _stepModuleMap():
    def getProp(prop):
        ret = []
        for i in _iterations:
            if i.name() == "ConvStep":
                continue
            ret.extend(getattr(i, prop)())
        return ret

    return collections.OrderedDict([
        ("ClusterMask", getProp("clusterMasking")),
        ("Seeding", getProp("seeding")),
        ("Building", getProp("building")),
        ("Fitting", getProp("fit")),
        ("Selection", getProp("selection")),
        ("Other", getProp("other"))
    ])

class TrackingTimingTable:
    def __init__(self):
        self._purpose = PlotPurpose.Timing
        self._page = "timing"
        self._section = "timing"

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self, dqmSubFolder):
        return self._section

    def create(self, tdirectory):
        h = tdirectory.Get("reconstruction_step_module_average")
        totalReco = None
        if h:
            totalReco = "%.1f" % h.Integral()

        creator = AggregateBins("iteration", "reconstruction_step_module_average", _iterModuleMap(includeConvStep=False), ignoreMissingBins=True)
        h = creator.create(tdirectory)
        totalTracking = None
        if h:
            totalTracking = "%.1f" % h.Integral()

        creator = AggregateBins("iteration", "reconstruction_step_module_average", _iterModuleMap(onlyConvStep=True), ignoreMissingBins=True)
        h = creator.create(tdirectory)
        totalConvStep = None
        if h:
            totalConvStep = "%.1f" % h.Integral()

        return [
            totalReco,
            totalTracking,
            totalConvStep,
        ]

    def headers(self):
        return [
            "Average reco time / event (ms)",
            "Average tracking (w/o convStep) time / event (ms)",
            "Average convStep time / event (ms)",
        ]

_common = {
    "drawStyle": "P",
    "xbinlabelsize": 10,
    "xbinlabeloption": "d"
}
_timing_summary = PlotGroup("summary", [
    Plot(AggregateBins("iteration", "reconstruction_step_module_average", _iterModuleMap(), ignoreMissingBins=True),
         ytitle="Average processing time (ms)", title="Average processing time / event", legendDx=-0.4, **_common),
    Plot(AggregateBins("iteration_fraction", "reconstruction_step_module_average", _iterModuleMap(), ignoreMissingBins=True),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
    #
    Plot(AggregateBins("step", "reconstruction_step_module_average", _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Average processing time (ms)", title="Average processing time / event", **_common),
    Plot(AggregateBins("step_fraction", "reconstruction_step_module_average", _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
#    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap), ytitle="Average processing time", title="Average processing time / event (normalized)", drawStyle="HIST", xbinlabelsize=0.03, normalizeToUnitArea=True)
#    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap, normalizeTo="ak7CaloJets"), ytitle="Average processing time / ak7CaloJets", title="Average processing time / event (normalized to ak7CaloJets)", drawStyle="HIST", xbinlabelsize=0.03)

    ],
                    legendDy=_legendDy_2rows
)
_timing_iterations = PlotGroup("iterations", [
    Plot(AggregateBins(i.name(), "reconstruction_step_module_average", collections.OrderedDict(i.modules()), ignoreMissingBins=True),
         ytitle="Average processing time (ms)", title=i.name(), **_common)
    for i in _iterations
],
                               legend=False
)
_pixelTiming = PlotGroup("pixelTiming", [
    Plot(AggregateBins("pixel", "reconstruction_step_module_average", {"pixelTracks": ["pixelTracks"]}), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST")
])

_timeFolders = [
    "DQMData/Run 1/DQM/Run summary/TimerService/Paths",
    "DQMData/Run 1/DQM/Run summary/TimerService/process RECO/Paths",
]
timePlotter = Plotter()
timePlotter.append("timing", _timeFolders, PlotFolder(
    _timing_summary,
    _timing_iterations,
    # _pixelTiming,
    loopSubFolders=False, purpose=PlotPurpose.Timing, page="timing"
))
timePlotter.appendTable("timing", _timeFolders, TrackingTimingTable())

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


