from __future__ import absolute_import
from builtins import range
import os
import copy
import collections

import six
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from .plotting import Subtract, FakeDuplicate, CutEfficiency, Transform, AggregateBins, ROC, Plot, PlotEmpty, PlotGroup, PlotOnSideGroup, PlotFolder, Plotter
from .html import PlotPurpose
from . import plotting
from . import validation
from . import html

########################################
#
# Per track collection plots
#
########################################

_maxEff = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025, 1.2, 1.5, 2]
_maxFake = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]

#_minMaxResol = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1]
_minMaxResol = [1e-5, 4e-5, 1e-4, 4e-4, 1e-3, 4e-3, 1e-2, 4e-2, 0.1, 0.4, 1]
_minMaxN = [5e-1, 5, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6, 5e7, 5e8, 5e9]

_minHits = [0, 5, 10]
_maxHits = [5, 10, 20, 40, 60, 80]
_minLayers = [0, 5, 10]
_maxLayers = [5, 10, 25]
_maxPixelLayers = 8
_min3DLayers = [0, 5, 10]
_max3DLayers = [5, 10, 20]
_minZ = [-60, -40, -20, -10, -5]
_maxZ = [5, 10, 20, 40, 60]
_minPU = [0, 10, 20, 50, 100, 150]
_maxPU = [20, 50, 65, 80, 100, 150, 200, 250]
_minMaxTracks = [0, 200, 500, 1000, 1500, 2000]
_minMaxMVA = [-1.025, -0.5, 0, 0.5, 1.025]

def _minMaxResidual(ma):
    return ([-x for x in ma], ma)

(_minResidualPhi, _maxResidualPhi) = _minMaxResidual([1e-4, 2e-4]) # rad
(_minResidualCotTheta, _maxResidualCotTheta) = _minMaxResidual([1e-4, 2e-4])
(_minResidualDxy, _maxResidualDxy) = _minMaxResidual([10, 20, 50, 100]) # um
(_minResidualDz, _maxResidualDz) = (_minResidualDxy, _maxResidualDxy)
(_minResidualPt, _maxResidualPt) = _minMaxResidual([1, 1.5, 2, 5]) # %


_legendDy_1row = 0.46
_legendDy_2rows = -0.025
_legendDy_2rows_3cols = -0.17
_legendDy_4rows = 0.09

# Ok, because of the way the "reference release" histograms are
# stored, this histogram is not accessible for the summary plots. To
# be fixed later with (large) reorganization of how things work.
#_trackingNumberOfEventsHistogram = "DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/tracks"

_trackingIterationOrderHistogram = "DQMData/Run 1/Tracking/Run summary/TrackBuilding/num_reco_coll"

def _makeEffFakeDupPlots(postfix, quantity, unit="", common={}, effopts={}, fakeopts={}):
    p = postfix
    q = quantity
    xq = q
    if unit != "":
        xq += " (" + unit + ")"

    effargs  = dict(xtitle="TP "+xq   , ytitle="efficiency vs "+q          , ymax=_maxEff)
    fakeargs = dict(xtitle="track "+xq, ytitle="fake+duplicates rate vs "+q, ymax=_maxFake)
    effargs.update(common)
    fakeargs.update(common)
    effargs.update(effopts)
    fakeargs.update(fakeopts)

    return [
        Plot("effic_vs_"+p, **effargs),
        Plot(FakeDuplicate("fakeduprate_vs_"+p, assoc="num_assoc(recoToSim)_"+p, dup="num_duplicate_"+p, reco="num_reco_"+p, title="fake+duplicates vs "+q), **fakeargs)
    ]

def _makeFakeDupPileupPlots(postfix, quantity, unit="", xquantity="", xtitle=None, common={}):
    p = postfix
    q = quantity
    if xtitle is None:
        if xquantity != "":
            xq = xquantity
        else:
            xq = q
            if unit != "":
                xq += " (" + unit + ")"
        xtitle="track "+xq

    return [
        Plot("fakerate_vs_"+p   , xtitle=xtitle, ytitle="fakerate vs "+q       , ymax=_maxFake, **common),
        Plot("duplicatesRate_"+p, xtitle=xtitle, ytitle="duplicates rate vs "+q, ymax=_maxFake, **common),
        Plot("pileuprate_"+p    , xtitle=xtitle, ytitle="pileup rate vs "+q    , ymax=_maxFake, **common),
    ]

def _makeFakeDist(postfix):
    return Subtract("num_fake_"+postfix, "num_reco_"+postfix, "num_assoc(recoToSim)_"+postfix)

def _makeDistPlots(postfix, quantity, common={}):
    p = postfix
    q = quantity

    args = dict(xtitle="track "+q, ylog=True, ymin=_minMaxN, ymax=_minMaxN)
    args.update(common)

    return [
        Plot("num_reco_"+p            , ytitle="tracks", **args),
        Plot("num_assoc(recoToSim)_"+p, ytitle="true tracks", **args),
        Plot(_makeFakeDist(p), ytitle="fake tracks", **args),
        Plot("num_duplicate_"+p       , ytitle="duplicate tracks", **args),
    ]

def _makeDistSimPlots(postfix, quantity, common={}):
    p = postfix
    q = quantity

    args = dict(xtitle="TP "+q, ylog=True, ymin=_minMaxN, ymax=_minMaxN)
    args.update(common)

    return [
        Plot("num_simul_"+p            , ytitle="TrackingParticles", **args),
        Plot("num_assoc(simToReco)_"+p, ytitle="Reconstructed TPs", **args),
    ]

def _makeMVAPlots(num, hp=False):
    pfix = "_hp" if hp else ""
    pfix2 = "Hp" if hp else ""

    xtitle = "MVA%d output"%num
    xtitlecut = "Cut on MVA%d output"%num
    args = dict(xtitle=xtitle, ylog=True, ymin=_minMaxN, ymax=_minMaxN)

    argsroc = dict(
        xtitle="Efficiency (excl. trk eff)", ytitle="Fake rate",
        xmax=_maxEff, ymax=_maxFake,
        drawStyle="EP",
    )
    argsroc2 = dict(
        ztitle="Cut on MVA%d"%num,
        xtitleoffset=5, ytitleoffset=6.5, ztitleoffset=4,
        adjustMarginRight=0.12
    )
    argsroc2.update(argsroc)
    argsroc2["drawStyle"] = "pcolz"

    argsprofile = dict(ymin=_minMaxMVA, ymax=_minMaxMVA)

    true_cuteff = CutEfficiency("trueeff_vs_mva%dcut%s"%(num,pfix), "num_assoc(recoToSim)_mva%dcut%s"%(num,pfix))
    fake_cuteff = CutEfficiency("fakeeff_vs_mva%dcut%s"%(num,pfix), Subtract("num_fake_mva%dcut%s"%(num,pfix), "num_reco_mva%dcut%s"%(num,pfix), "num_assoc(recoToSim)_mva%dcut%s"%(num,pfix)))

    return [
        PlotGroup("mva%d%s"%(num,pfix2), [
            Plot("num_assoc(recoToSim)_mva%d%s"%(num,pfix), ytitle="true tracks", **args),
            Plot(Subtract("num_fake_mva%d%s"%(num,pfix), "num_reco_mva%d%s"%(num,pfix), "num_assoc(recoToSim)_mva%d%s"%(num,pfix)), ytitle="fake tracks", **args),
            Plot("effic_vs_mva%dcut%s"%(num,pfix), xtitle=xtitlecut, ytitle="Efficiency (excl. trk eff)", ymax=_maxEff),
            #
            Plot("fakerate_vs_mva%dcut%s"%(num,pfix), xtitle=xtitlecut, ytitle="Fake rate", ymax=_maxFake),
            Plot(ROC("effic_vs_fake_mva%d%s"%(num,pfix), "effic_vs_mva%dcut%s"%(num,pfix), "fakerate_vs_mva%dcut%s"%(num,pfix)), **argsroc),
            Plot(ROC("effic_vs_fake_mva%d%s"%(num,pfix), "effic_vs_mva%dcut%s"%(num,pfix), "fakerate_vs_mva%dcut%s"%(num,pfix), zaxis=True), **argsroc2),
            # Same signal efficiency, background efficiency, and ROC definitions as in TMVA
            Plot(true_cuteff, xtitle=xtitlecut, ytitle="True track selection efficiency", ymax=_maxEff),
            Plot(fake_cuteff, xtitle=xtitlecut, ytitle="Fake track selection efficiency", ymax=_maxEff),
            Plot(ROC("true_eff_vs_fake_rej_mva%d%s"%(num,pfix), true_cuteff, Transform("fake_rej_mva%d%s"%(num,pfix), fake_cuteff, lambda x: 1-x)), xtitle="True track selection efficiency", ytitle="Fake track rejection", xmax=_maxEff, ymax=_maxEff),
        ], ncols=3, legendDy=_legendDy_1row),
        PlotGroup("mva%d%sPtEta"%(num,pfix2), [
            Plot("mva_assoc(recoToSim)_mva%d%s_pT"%(num,pfix), xtitle="Track p_{T} (GeV)", ytitle=xtitle+" for true tracks", xlog=True, **argsprofile),
            Plot("mva_fake_mva%d%s_pT"%(num,pfix), xtitle="Track p_{T} (GeV)", ytitle=xtitle+" for fake tracks", xlog=True, **argsprofile),
            Plot("mva_assoc(recoToSim)_mva%d%s_eta"%(num,pfix), xtitle="Track #eta", ytitle=xtitle+" for true tracks", **argsprofile),
            Plot("mva_fake_mva%d%s_eta"%(num,pfix), xtitle="Track #eta", ytitle=xtitle+" for fake tracks", **argsprofile),
        ], legendDy=_legendDy_2rows)
    ]

_effandfakePtEtaPhi = PlotGroup("effandfakePtEtaPhi", [
    Plot("efficPt", title="Efficiency vs p_{T}", xtitle="TP p_{T} (GeV)", ytitle="efficiency vs p_{T}", xlog=True, ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_pT", assoc="num_assoc(recoToSim)_pT", dup="num_duplicate_pT", reco="num_reco_pT", title="fake+duplicates vs p_{T}"),
         xtitle="track p_{T} (GeV)", ytitle="fake+duplicates rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("effic", xtitle="TP #eta", ytitle="efficiency vs #eta", title="", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_eta", assoc="num_assoc(recoToSim)_eta", dup="num_duplicate_eta", reco="num_reco_eta", title=""),
         xtitle="track #eta", ytitle="fake+duplicates rate vs #eta", ymax=_maxFake),
] +
    _makeEffFakeDupPlots("phi", "#phi")
)

_effandfakeDxyDzBS = PlotGroup("effandfakeDxyDzBS",
                               _makeEffFakeDupPlots("dxy"  , "dxy"    , "cm") +
                               _makeEffFakeDupPlots("dz"   , "dz"     , "cm"),
                               legendDy=_legendDy_2rows
)
_effandfakeDxyDzPV = PlotGroup("effandfakeDxyDzPV",
                               _makeEffFakeDupPlots("dxypv"       , "dxy(PV)", "cm") +
                               _makeEffFakeDupPlots("dxypv_zoomed", "dxy(PV)", "cm") +
                               _makeEffFakeDupPlots("dzpv"        , "dz(PV)" , "cm") +
                               _makeEffFakeDupPlots("dzpv_zoomed" , "dz(PV)" , "cm"),
                               legendDy=_legendDy_4rows
)
_effandfakeHitsLayers = PlotGroup("effandfakeHitsLayers",
                                  _makeEffFakeDupPlots("hit"       , "hits"        , common=dict(xmin=_minHits    , xmax=_maxHits)) +
                                  _makeEffFakeDupPlots("layer"     , "layers"      , common=dict(xmin=_minLayers  , xmax=_maxLayers)) +
                                  _makeEffFakeDupPlots("pixellayer", "pixel layers", common=dict(                   xmax=_maxPixelLayers)) +
                                  _makeEffFakeDupPlots("3Dlayer"   , "3D layers"   , common=dict(xmin=_min3DLayers, xmax=_max3DLayers)),
                                  legendDy=_legendDy_4rows
)
_common = {"ymin": 0, "ymax": _maxEff}
_effandfakePos = PlotGroup("effandfakePos",
                           _makeEffFakeDupPlots("vertpos", "vert r", "cm", fakeopts=dict(xtitle="track ref. point r (cm)", ytitle="fake+duplicates vs. r"), common=dict(xlog=True)) +
                           _makeEffFakeDupPlots("zpos"   , "vert z", "cm", fakeopts=dict(xtitle="track ref. point z (cm)", ytitle="fake+duplicates vs. z")) +
                           _makeEffFakeDupPlots("simpvz" , "Sim. PV z", "cm", common=dict(xtitle="Sim. PV z (cm)", xmin=_minZ, xmax=_maxZ))
)
_effandfakeDeltaRPU = PlotGroup("effandfakeDeltaRPU",
                                _makeEffFakeDupPlots("dr"     , "#DeltaR", effopts=dict(xtitle="TP min #DeltaR"), fakeopts=dict(xtitle="track min #DeltaR"), common=dict(xlog=True)) +
                                _makeEffFakeDupPlots("drj" , "#DeltaR(track, jet)", effopts=dict(xtitle="#DeltaR(TP, jet)", ytitle="efficiency vs #DeltaR(TP, jet"), fakeopts=dict(xtitle="#DeltaR(track, jet)"), common=dict(xlog=True))+
                                _makeEffFakeDupPlots("pu"     , "PU"     , common=dict(xtitle="Pileup", xmin=_minPU, xmax=_maxPU)),
                                legendDy=_legendDy_4rows
)


_algos_common = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d")
_duplicateAlgo = PlotOnSideGroup("duplicateAlgo", Plot("duplicates_oriAlgo_vs_oriAlgo", drawStyle="COLZ", adjustMarginLeft=0.1, adjustMarginRight=0.1, **_algos_common))

_dupandfakePtEtaPhi = PlotGroup("dupandfakePtEtaPhi", [
    Plot("fakeratePt", xtitle="track p_{T} (GeV)", ytitle="fakerate vs p_{T}", xlog=True, ymax=_maxFake),
    Plot("duplicatesRate_Pt", xtitle="track p_{T} (GeV)", ytitle="duplicates rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("pileuprate_Pt", xtitle="track p_{T} (GeV)", ytitle="pileup rate vs p_{T}", ymax=_maxFake, xlog=True),
    Plot("fakerate", xtitle="track #eta", ytitle="fakerate vs #eta", title="", ymax=_maxFake),
    Plot("duplicatesRate", xtitle="track #eta", ytitle="duplicates rate vs #eta", title="", ymax=_maxFake),
    Plot("pileuprate", xtitle="track #eta", ytitle="pileup rate vs #eta", title="", ymax=_maxFake),
] + _makeFakeDupPileupPlots("phi", "#phi"),
                         ncols=3
)
_dupandfakeDxyDzBS = PlotGroup("dupandfakeDxyDzBS",
                               _makeFakeDupPileupPlots("dxy"  , "dxy"    , "cm") +
                               _makeFakeDupPileupPlots("dz"   , "dz"     , "cm"),
                               ncols=3, legendDy=_legendDy_2rows_3cols
)
_dupandfakeDxyDzPV = PlotGroup("dupandfakeDxyDzPV",
                               _makeFakeDupPileupPlots("dxypv"       , "dxy(PV)", "cm") +
                               _makeFakeDupPileupPlots("dxypv_zoomed", "dxy(PV)", "cm") +
                               _makeFakeDupPileupPlots("dzpv"        , "dz(PV)" , "cm") +
                               _makeFakeDupPileupPlots("dzpv_zoomed" , "dz(PV)" , "cm"),
                               ncols=3, legendDy=_legendDy_4rows
)
_dupandfakeHitsLayers = PlotGroup("dupandfakeHitsLayers",
                                  _makeFakeDupPileupPlots("hit"       , "hits"        , common=dict(xmin=_minHits    , xmax=_maxHits)) +
                                  _makeFakeDupPileupPlots("layer"     , "layers"      , common=dict(xmin=_minLayers  , xmax=_maxLayers)) +
                                  _makeFakeDupPileupPlots("pixellayer", "pixel layers", common=dict(                   xmax=_maxPixelLayers)) +
                                  _makeFakeDupPileupPlots("3Dlayer"   , "3D layers"   , common=dict(xmin=_min3DLayers, xmax=_max3DLayers)),
                                  ncols=3, legendDy=_legendDy_4rows
)
_dupandfakePos = PlotGroup("dupandfakePos",
                           _makeFakeDupPileupPlots("vertpos", "r", "cm", xquantity="ref. point r (cm)", common=dict(xlog=True)) +
                           _makeFakeDupPileupPlots("zpos"   , "z", "cm", xquantity="ref. point z (cm)") +
                           _makeFakeDupPileupPlots("simpvz" , "Sim. PV z", xtitle="Sim. PV z (cm)", common=dict(xmin=_minZ, xmax=_maxZ)),
                           ncols=3,
)
_dupandfakeDeltaRPU = PlotGroup("dupandfakeDeltaRPU",
                                _makeFakeDupPileupPlots("dr"     , "#DeltaR", xquantity="min #DeltaR", common=dict(xlog=True)) +
                                _makeFakeDupPileupPlots("drj"     , "#DeltaR(track, jet)", xtitle="#DeltaR(track, jet)", common=dict(xlog=True)) +
                                _makeFakeDupPileupPlots("pu"     , "PU"     , xtitle="Pileup", common=dict(xmin=_minPU, xmax=_maxPU)),
                                ncols=3
)
_seedingLayerSet_common = dict(removeEmptyBins=True, xbinlabelsize=8, xbinlabeloption="d", adjustMarginRight=0.1)
_dupandfakeSeedingPlots = _makeFakeDupPileupPlots("seedingLayerSet", "seeding layers", xtitle="", common=_seedingLayerSet_common)
_dupandfakeChi2Seeding = PlotGroup("dupandfakeChi2Seeding",
                                   _makeFakeDupPileupPlots("chi2", "#chi^{2}") +
                                   _dupandfakeSeedingPlots,
                                   ncols=3, legendDy=_legendDy_2rows_3cols
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
    Plot("missing_inner_layers", xmin=_minLayers, xmax=_maxLayers, ymax=1, **_common),
    Plot("missing_outer_layers", xmin=_minLayers, xmax=_maxLayers, ymax=1, **_common),
    Plot("hits_eta", xtitle="track #eta", ytitle="<hits> vs #eta", ymin=_minHits, ymax=_maxHits, statyadjust=[0,0,-0.15],
         fallback={"name": "nhits_vs_eta", "profileX": True}),
    Plot("hits", stat=True, xtitle="track hits", xmin=_minHits, xmax=_maxHits, ylog=True, ymin=[5e-1, 5, 5e1, 5e2, 5e3], drawStyle="hist"),
    Plot("num_simul_pT", xtitle="TP p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common),
    Plot("num_reco_pT", xtitle="track p_{T}", xlog=True, ymax=[1e-1, 2e-1, 5e-1, 1], **_common)
])
_tuning = PlotGroup("tuning", [
    Plot("chi2", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=[0.1, 0.2, 0.5, 1.0001], drawStyle="hist", xtitle="#chi^{2}", ratioUncertainty=False),
    Plot("chi2_prob", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="Prob(#chi^{2})"),
    Plot("chi2mean", title="", xtitle="#eta", ytitle="< #chi^{2} / ndf >", ymin=[0, 0.5], ymax=[2, 2.5, 3, 5],
         fallback={"name": "chi2_vs_eta", "profileX": True}),
    Plot("ptres_vs_eta_Mean", scale=100, title="", xtitle="TP #eta (PCA to beamline)", ytitle="< #delta p_{T} / p_{T} > (%)", ymin=_minResidualPt, ymax=_maxResidualPt),
    Plot("chi2mean_vs_pt", title="", xtitle="p_{T}", ytitle="< #chi^{2} / ndf >", ymin=[0, 0.5], ymax=[2, 2.5, 3, 5], xlog=True, fallback={"name": "chi2_vs_pt", "profileX": True}),
    Plot("ptres_vs_pt_Mean", title="", xtitle="p_{T}", ytitle="< #delta p_{T}/p_{T} > (%)", scale=100, ymin=_minResidualPt, ymax=_maxResidualPt,xlog=True)
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
_common = {"title": "", "ylog": True, "xtitle": "TP #eta (PCA to beamline)", "ymin": _minMaxResol, "ymax": _minMaxResol}
_resolutionsEta = PlotGroup("resolutionsEta", [
    Plot("phires_vs_eta_Sigma", ytitle="#sigma(#delta #phi) (rad)", **_common),
    Plot("cotThetares_vs_eta_Sigma", ytitle="#sigma(#delta cot(#theta))", **_common),
    Plot("dxyres_vs_eta_Sigma", ytitle="#sigma(#delta d_{xy}) (cm)", **_common),
    Plot("dzres_vs_eta_Sigma", ytitle="#sigma(#delta d_{z}) (cm)", **_common),
    Plot("ptres_vs_eta_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", **_common),
])
_common = {"title": "", "ylog": True, "xlog": True, "xtitle": "TP p_{T} (PCA to beamline)", "xmin": 0.1, "xmax": 1000, "ymin": _minMaxResol, "ymax": _minMaxResol}
_resolutionsPt = PlotGroup("resolutionsPt", [
    Plot("phires_vs_pt_Sigma", ytitle="#sigma(#delta #phi) (rad)", **_common),
    Plot("cotThetares_vs_pt_Sigma", ytitle="#sigma(#delta cot(#theta))", **_common),
    Plot("dxyres_vs_pt_Sigma", ytitle="#sigma(#delta d_{xy}) (cm)", **_common),
    Plot("dzres_vs_pt_Sigma", ytitle="#sigma(#delta d_{z}) (cm)", **_common),
    Plot("ptres_vs_pt_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", **_common),
])

## Extended set of plots
_extDistPtEtaPhi = PlotGroup("distPtEtaPhi",
                             _makeDistPlots("pT", "p_{T} (GeV)", common=dict(xlog=True)) +
                             _makeDistPlots("eta", "#eta") +
                             _makeDistPlots("phi", "#phi"),
                             ncols=4)
_extDistDxyDzBS = PlotGroup("distDxyDzBS",
                            _makeDistPlots("dxy"  , "dxy (cm)") +
                            _makeDistPlots("dz"   , "dz (cm)"),
                            ncols=4, legendDy=_legendDy_2rows)
_extDistDxyDzPV = PlotGroup("distDxyDzPV",
                            _makeDistPlots("dxypv"       , "dxy(PV) (cm)") +
                            _makeDistPlots("dxypv_zoomed", "dxy(PV) (cm)") +
                            _makeDistPlots("dzpv"        , "dz(PV) (cm)") +
                            _makeDistPlots("dzpv_zoomed" , "dz(PV) (cm)"),
                            ncols=4, legendDy=_legendDy_4rows)
_extDistHitsLayers = PlotGroup("distHitsLayers",
                               _makeDistPlots("hit"       , "hits"        , common=dict(xmin=_minHits    , xmax=_maxHits)) +
                               _makeDistPlots("layer"     , "layers"      , common=dict(xmin=_minLayers  , xmax=_maxLayers)) +
                               _makeDistPlots("pixellayer", "pixel layers", common=dict(                   xmax=_maxPixelLayers)) +
                               _makeDistPlots("3Dlayer"   , "3D layers"   , common=dict(xmin=_min3DLayers, xmax=_max3DLayers)),
                               ncols=4, legendDy=_legendDy_4rows,
)
_extDistPos = PlotGroup("distPos",
                              _makeDistPlots("vertpos", "ref. point r (cm)", common=dict(xlog=True)) +
                              _makeDistPlots("zpos"   , "ref. point z (cm)") +
                              _makeDistPlots("simpvz" , "Sim. PV z (cm)", common=dict(xmin=_minZ, xmax=_maxZ)),
                              ncols=3,
)
_extDistDeltaR = PlotGroup("distDeltaR",
                              _makeDistPlots("dr"     , "min #DeltaR", common=dict(xlog=True)) +
                              _makeDistPlots("drj"     , "#DeltaR(track, jet)", common=dict(xlog=True)),
                              ncols=2, legendDy=_legendDy_2rows,
)
_extDistSeedingPlots = _makeDistPlots("seedingLayerSet", "seeding layers", common=dict(xtitle="", **_seedingLayerSet_common))
_extDistChi2Seeding = PlotGroup("distChi2Seeding",
                                _makeDistPlots("chi2", "#chi^{2}") +
                                _extDistSeedingPlots,
                                ncols=4, legendDy=_legendDy_2rows_3cols
)
_common = dict(title="", xtitle="TP #eta (PCA to beamline)")
_extResidualEta = PlotGroup("residualEta", [
    Plot("phires_vs_eta_Mean", ytitle="< #delta #phi > (rad)", ymin=_minResidualPhi, ymax=_maxResidualPhi, **_common),
    Plot("cotThetares_vs_eta_Mean", ytitle="< #delta cot(#theta) >", ymin=_minResidualCotTheta, ymax=_maxResidualCotTheta, **_common),
    Plot("dxyres_vs_eta_Mean", ytitle="< #delta d_{xy} > (#mum)", scale=10000, ymin=_minResidualDxy, ymax=_maxResidualDxy, **_common),
    Plot("dzres_vs_eta_Mean", ytitle="< #delta d_{z} > (#mum)", scale=10000, ymin=_minResidualDz, ymax=_maxResidualDz, **_common),
    Plot("ptres_vs_eta_Mean", ytitle="< #delta p_{T}/p_{T} > (%)", scale=100, ymin=_minResidualPt, ymax=_maxResidualPt, **_common), # same as in tuning, but to be complete
])
_common = dict(title="", xlog=True, xtitle="TP p_{T} (PCA to beamline)", xmin=0.1, xmax=1000)
_extResidualPt = PlotGroup("residualPt", [
    Plot("phires_vs_pt_Mean", ytitle="< #delta #phi > (rad)", ymin=_minResidualPhi, ymax=_maxResidualPhi, **_common),
    Plot("cotThetares_vs_pt_Mean", ytitle="< #delta cot(#theta > )", ymin=_minResidualCotTheta, ymax=_maxResidualCotTheta, **_common),
    Plot("dxyres_vs_pt_Mean", ytitle="< #delta d_{xy} > (#mum)", scale=10000, ymin=_minResidualDxy, ymax=_maxResidualDxy, **_common),
    Plot("dzres_vs_pt_Mean", ytitle="< #delta d_{z} > (#mum)", scale=10000, ymin=_minResidualDz, ymax=_maxResidualDz, **_common),
    Plot("ptres_vs_pt_Mean", ytitle="< #delta p_{T}/p_{T} > (%)", scale=100, ymin=_minResidualPt, ymax=_maxResidualPt, **_common), # same as in tuning, but to be complete
])
_common = dict(title="", ytitle="Selected tracks/TrackingParticles", ymax=_maxEff)
_extNrecVsNsim = PlotGroup("nrecVsNsim", [
    Plot("nrec_vs_nsim", title="", xtitle="TrackingParticles", ytitle="Tracks", profileX=True, xmin=_minMaxTracks, xmax=_minMaxTracks, ymin=_minMaxTracks, ymax=_minMaxTracks),
    Plot("nrecPerNsim_vs_pu", xtitle="Pileup", xmin=_minPU, xmax=_maxPU, **_common),
    Plot("nrecPerNsimPt", xtitle="p_{T} (GeV)", xlog=True, **_common),
    Plot("nrecPerNsim", xtitle="#eta", **_common)
], legendDy=_legendDy_2rows)
_extHitsLayers = PlotGroup("hitsLayers", [
    Plot("PXLhits_vs_eta", xtitle="#eta", ytitle="<pixel hits>"),
    Plot("PXLlayersWithMeas_vs_eta", xtitle="#eta", ytitle="<pixel layers>"),
    Plot("STRIPhits_vs_eta", xtitle="#eta", ytitle="<strip hits>"),
    Plot("STRIPlayersWithMeas_vs_eta", xtitle="#eta", ytitle="<strip layers>"),
], legendDy=_legendDy_2rows)


## Extended set of plots also for simulation
_extDistSimPtEtaPhi = PlotGroup("distsimPtEtaPhi",
                                _makeDistSimPlots("pT", "p_{T} (GeV)", common=dict(xlog=True)) +
                                _makeDistSimPlots("eta", "#eta") +
                                _makeDistSimPlots("phi", "#phi"),
                                ncols=2)
_extDistSimDxyDzBS = PlotGroup("distsimDxyDzBS",
                               _makeDistSimPlots("dxy"  , "dxy (cm)") +
                               _makeDistSimPlots("dz"   , "dz (cm)"),
                               ncols=2, legendDy=_legendDy_2rows)
_extDistSimDxyDzPV = PlotGroup("distsimDxyDzPV",
                               _makeDistSimPlots("dxypv"       , "dxy(PV) (cm)") +
                               _makeDistSimPlots("dxypv_zoomed", "dxy(PV) (cm)") +
                               _makeDistSimPlots("dzpv"        , "dz(PV) (cm)") +
                               _makeDistSimPlots("dzpv_zoomed" , "dz(PV) (cm)"),
                               ncols=2, legendDy=_legendDy_4rows)
_extDistSimHitsLayers = PlotGroup("distsimHitsLayers",
                                  _makeDistSimPlots("hit"       , "hits"        , common=dict(xmin=_minHits    , xmax=_maxHits)) +
                                  _makeDistSimPlots("layer"     , "layers"      , common=dict(xmin=_minLayers  , xmax=_maxLayers)) +
                                  _makeDistSimPlots("pixellayer", "pixel layers", common=dict(                   xmax=_maxPixelLayers)) +
                                  _makeDistSimPlots("3Dlayer"   , "3D layers"   , common=dict(xmin=_min3DLayers, xmax=_max3DLayers)),
                                  ncols=2, legendDy=_legendDy_4rows,
)
_extDistSimPos = PlotGroup("distsimPos",
                                 _makeDistSimPlots("vertpos", "vert r (cm)", common=dict(xlog=True)) +
                                 _makeDistSimPlots("zpos"   , "vert z (cm)") +
                                 _makeDistSimPlots("simpvz" , "Sim. PV z (cm)", common=dict(xmin=_minZ, xmax=_maxZ)),
                                 ncols=3,
)
_extDistSimDeltaR = PlotGroup("distsimDeltaR",
                                 _makeDistSimPlots("dr"     , "min #DeltaR", common=dict(xlog=True)) +
                                 _makeDistSimPlots("drj" , "#DeltaR(TP, jet)", common=dict(xlog=True)),
                                 ncols=2, legendDy=_legendDy_2rows,
)

########################################
#
# Summary plots
#
########################################

_possibleTrackingNonIterationColls = [
    'ak4PFJets',
    'btvLike',
]
_possibleTrackingColls = [
    'initialStepPreSplitting',
    'initialStep',
    'highPtTripletStep', # phase1
    'detachedQuadStep', # phase1
    'detachedTripletStep',
    'lowPtQuadStep', # phase1
    'lowPtTripletStep',
    'pixelPairStepA', # seeds
    'pixelPairStepB', # seeds
    'pixelPairStepC', # seeds
    'pixelPairStep',
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
] + _possibleTrackingNonIterationColls
_possibleTrackingCollsOld = {
    "Zero"  : "iter0",
    "First" : "iter1",
    "Second": "iter2",
    "Third" : "iter3",
    "Fourth": "iter4",
    "Fifth" : "iter5",
    "Sixth" : "iter6",
    "Seventh": "iter7",
    "Ninth" : "iter9",
    "Tenth" : "iter10",
}

def _trackingSubFoldersFallbackSLHC_Phase1PU140(subfolder):
    ret = subfolder.replace("trackingParticleRecoAsssociation", "AssociatorByHitsRecoDenom")
    for (old, new) in [("InitialStep",         "Zero"),
                       ("HighPtTripletStep",   "First"),
                       ("LowPtQuadStep",       "Second"),
                       ("LowPtTripletStep",    "Third"),
                       ("DetachedQuadStep",    "Fourth"),
                       ("PixelPairStep",       "Fifth"),
                       ("MuonSeededStepInOut", "Ninth"),
                       ("MuonSeededStepOutIn", "Tenth")]:
        ret = ret.replace(old, new)
    if ret == subfolder:
        return None
    return ret
def _trackingRefFileFallbackSLHC_Phase1PU140(path):
    for (old, new) in [("initialStep",         "iter0"),
                       ("highPtTripletStep",   "iter1"),
                       ("lowPtQuadStep",       "iter2"),
                       ("lowPtTripletStep",    "iter3"),
                       ("detachedQuadStep",    "iter4"),
                       ("pixelPairStep",       "iter5"),
                       ("muonSeededStepInOut", "iter9"),
                       ("muonSeededStepOutIn", "iter10")]:
        path = path.replace(old, new)
    return path

def _trackingSubFoldersFallbackFromPV(subfolder):
    return subfolder.replace("trackingParticleRecoAsssociation", "trackingParticleRecoAsssociationSignal")
def _trackingSubFoldersFallbackConversion(subfolder):
    return subfolder.replace("quickAssociatorByHits", "quickAssociatorByHitsConversion")
def _trackingSubFoldersFallbackPreSplitting(subfolder):
    return subfolder.replace("quickAssociatorByHits", "quickAssociatorByHitsPreSplitting")

# Additional "quality" flags than highPurity. In a separate list to
# allow customization.
_additionalTrackQualities = [
    "Pt09",
    "ByOriginalAlgo",
    "ByAlgoMask"
]
def _mapCollectionToAlgoQuality(collName):
    if "Hp" in collName:
        quality = "highPurity"
    else:
        quality = ""
    collNameNoQuality = collName.replace("Hp", "")
    for qual in _additionalTrackQualities:
        if qual in collName:
            quality += qual
            collNameNoQuality = collNameNoQuality.replace(qual, "")

    collNameNoQuality = collNameNoQuality.replace("Tracks", "", 1) # make summary naming consistent with iteration folders
    collNameLow = collNameNoQuality.lower().replace("frompv2", "").replace("frompv", "").replace("frompvalltp", "").replace("alltp", "")

    if collNameLow.find("seed") == 0:
        collNameLow = collNameLow[4:]
        if collNameLow == "initialstepseedspresplitting":
            collNameLow = "initialsteppresplittingseeds"
        elif collNameLow == "muonseededseedsinout":
            collNameLow = "muonseededstepinoutseeds"
        elif collNameLow == "muonseededseedsoutin":
            collNameLow = "muonseededstepoutinseeds"

        i_seeds = collNameLow.index("seeds")
        quality = collNameLow[i_seeds:]+quality

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
            for coll, name in six.iteritems(_possibleTrackingCollsOld):
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
_collLabelMapHp = collections.OrderedDict(map(_collhelper, ["generalTracks"]+[n for n in _possibleTrackingColls if "Step" in n]))
def _summaryBinRename(binLabel, highPurity, byOriginalAlgo, byAlgoMask, ptCut, seeds):
    (algo, quality) = _mapCollectionToAlgoQuality(binLabel)
    if algo == "ootb":
        algo = "generalTracks"
    ret = None

    if byOriginalAlgo:
        if algo != "generalTracks" and "ByOriginalAlgo" not in quality: # keep generalTracks bin as well
            return None
        quality = quality.replace("ByOriginalAlgo", "")
    if byAlgoMask:
        if algo != "generalTracks" and "ByAlgoMask" not in quality: # keep generalTracks bin as well
            return None
        quality = quality.replace("ByAlgoMask", "")
    if ptCut:
        if "Pt09" not in quality:
            return None
        quality = quality.replace("Pt09", "")

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

def _constructSummary(mapping=None, highPurity=False, byOriginalAlgo=False, byAlgoMask=False, ptCut=False, seeds=False, midfix=""):
    _common = {"drawStyle": "EP", "xbinlabelsize": 10, "xbinlabeloption": "d"}
    _commonN = dict(ylog=True, ymin=_minMaxN, ymax=_minMaxN,
                    normalizeToNumberOfEvents=True,
    )
    _commonN.update(_common)
    _commonAB = dict(mapping=mapping,
                     renameBin=lambda bl: _summaryBinRename(bl, highPurity, byOriginalAlgo, byAlgoMask, ptCut, seeds),
                     ignoreMissingBins=True,
                     originalOrder=True,
    )
    if byOriginalAlgo or byAlgoMask:
        _commonAB["minExistingBins"] = 2
    prefix = "summary"+midfix

    h_eff = "effic_vs_coll"
    h_fakerate = "fakerate_vs_coll"
    h_duplicaterate = "duplicatesRate_coll"
    h_pileuprate = "pileuprate_coll"

    h_reco = "num_reco_coll"
    h_true = "num_assoc(recoToSim)_coll"
    h_fake = Subtract("num_fake_coll_orig", "num_reco_coll", "num_assoc(recoToSim)_coll")
    h_duplicate = "num_duplicate_coll"
    h_pileup = "num_pileup_coll"
    if mapping is not None:
        h_eff = AggregateBins("efficiency", h_eff, **_commonAB)
        h_fakerate = AggregateBins("fakerate", h_fakerate, **_commonAB)
        h_duplicaterate = AggregateBins("duplicatesRate", h_duplicaterate, **_commonAB)
        h_pileuprate = AggregateBins("pileuprate", h_pileuprate, **_commonAB)

        h_reco = AggregateBins("num_reco_coll", h_reco, **_commonAB)
        h_true = AggregateBins("num_true_coll", h_true, **_commonAB)
        h_fake = AggregateBins("num_fake_coll", h_fake, **_commonAB)
        h_duplicate = AggregateBins("num_duplicate_coll", h_duplicate, **_commonAB)
        h_pileup = AggregateBins("num_pileup_coll", h_pileup, **_commonAB)

    summary = PlotGroup(prefix, [
        Plot(h_eff, title="Efficiency vs collection", ytitle="Efficiency", ymin=1e-3, ymax=1, ylog=True, **_common),
        Plot(h_fakerate, title="Fakerate vs collection", ytitle="Fake rate", ymax=_maxFake, **_common),
        #
        Plot(h_duplicaterate, title="Duplicates rate vs collection", ytitle="Duplicates rate", ymax=_maxFake, **_common),
        Plot(h_pileuprate, title="Pileup rate vs collection", ytitle="Pileup rate", ymax=_maxFake, **_common),
        ],
                        legendDy=_legendDy_2rows
    )
    summaryN = PlotGroup(prefix+"_ntracks", [
        # FIXME
        #Plot(h_reco, ytitle="Tracks/event", title="Number of tracks/event vs collection", **_commonN),
        #Plot(h_true, ytitle="True tracks/event", title="Number of true tracks/event vs collection", **_commonN),
        #Plot(h_fake, ytitle="Fake tracks/event", title="Number of fake tracks/event vs collection", **_commonN),
        #Plot(h_duplicate, ytitle="Duplicate tracks/event", title="Number of duplicate tracks/event vs collection", **_commonN),
        #Plot(h_pileup, ytitle="Pileup tracks/event", title="Number of pileup tracks/event vs collection", **_commonN),
        Plot(h_reco, ytitle="Tracks", title="Number of tracks vs collection", **_commonN),
        Plot(h_true, ytitle="True tracks", title="Number of true tracks vs collection", **_commonN),
        Plot(h_fake, ytitle="Fake tracks", title="Number of fake tracks vs collection", **_commonN),
        Plot(h_duplicate, ytitle="Duplicate tracks", title="Number of duplicate tracks vs collection", **_commonN),
        Plot(h_pileup, ytitle="Pileup tracks", title="Number of pileup tracks vs collection", **_commonN),
    ])

    return (summary, summaryN)

(_summaryRaw,              _summaryRawN)              = _constructSummary(midfix="Raw")
(_summary,                 _summaryN)                 = _constructSummary(_collLabelMap)
(_summaryHp,               _summaryNHp)               = _constructSummary(_collLabelMapHp, highPurity=True)
(_summaryByOriginalAlgo,   _summaryByOriginalAlgoN)   = _constructSummary(_collLabelMapHp, byOriginalAlgo=True, midfix="ByOriginalAlgo")
(_summaryByOriginalAlgoHp, _summaryByOriginalAlgoNHp) = _constructSummary(_collLabelMapHp, byOriginalAlgo=True, midfix="ByOriginalAlgo", highPurity=True)
(_summaryByAlgoMask,       _summaryByAlgoMaskN)       = _constructSummary(_collLabelMapHp, byAlgoMask=True, midfix="ByAlgoMask")
(_summaryByAlgoMaskHp,     _summaryByAlgoMaskNHp)     = _constructSummary(_collLabelMapHp, byAlgoMask=True, midfix="ByAlgoMask", highPurity=True)
(_summaryPt09,             _summaryPt09N)             = _constructSummary(_collLabelMap, ptCut=True, midfix="Pt09")
(_summaryPt09Hp,           _summaryPt09NHp)           = _constructSummary(_collLabelMap, ptCut=True, midfix="Pt09", highPurity=True)
(_summarySeeds,            _summarySeedsN)            = _constructSummary(_collLabelMapHp, seeds=True)

########################################
#
# PackedCandidate plots
#
########################################

_common = {"normalizeToUnitArea": True, "ylog": True, "ymin": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "ymax": [1e-2, 1e-1, 1.1]}
_commonStatus = {}
_commonStatus.update(_common)
_commonStatus.update({"xbinlabelsize": 10, "xbinlabeloption": "d", "drawStyle": "hist", "adjustMarginRight": 0.08})
_commonLabelSize = {}
_commonLabelSize.update(_common)
_commonLabelSize.update({"xlabelsize": 17})

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

_packedCandidateLayers = PlotGroup("layers", [
    PlotEmpty(),
    Plot("diffNumberOfPixelLayers", xtitle="Pixel layers",  **_common),
    Plot("diffNumberOfStripLayers", xtitle="Strip layers", **_common),
    #
    Plot("diffHitPatternTrackerLayersWithMeasurement", xtitle="Layers (via HitPattern)", **_common),
    Plot("diffHitPatternPixelLayersWithMeasurement", xtitle="Pixel layers (via HitPattern)", **_common),
    Plot("diffHitPatternStripLayersWithMeasurement", xtitle="Strip layers (via HitPattern)", **_common),
    #
    Plot("numberLayersOverMax", xtitle="Number of overflown layers", **_common),
    Plot("numberPixelLayersOverMax", xtitle="Number of overflown pixel layers", **_common),
    Plot("numberStripLayersOverMax", xtitle="Number of overflown strip layers", **_common),
],
                                   ncols=3
)


_packedCandidateImpactParameter1 = PlotGroup("impactParameter1", [
    Plot("diffDxyAssocPV", xtitle="dxy(assocPV)", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffDxyAssocPVStatus", **_commonStatus),
    Plot("diffDxyAssocPVUnderOverFlowSign", xtitle="dxy(assocPV)", **_common),
    Plot("diffDzAssocPV", xtitle="dz(assocPV)", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffDzAssocPVStatus", **_commonStatus),
    Plot("diffDzAssocPVUnderOverFlowSign", xtitle="dz(assocPV)", **_common),
    Plot("diffDxyError", xtitle="dxyError()", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffDszError", xtitle="dszError()", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffDzError", xtitle="dzError()", adjustMarginRight=0.02, **_commonLabelSize),

],
                                             ncols=3
)

_packedCandidateImpactParameter2 = PlotGroup("impactParameter2", [
    Plot("diffDxyPV", xtitle="dxy(PV) via PC", **_commonLabelSize),
    Plot("diffDzPV", xtitle="dz(PV) via PC", **_commonLabelSize),
    Plot("diffTrackDxyAssocPV", xtitle="dxy(PV) via PC::bestTrack()", **_commonLabelSize),
    Plot("diffTrackDzAssocPV", xtitle="dz(PV) via PC::bestTrack()", **_commonLabelSize),
    Plot("diffTrackDxyError", xtitle="dxyError() via PC::bestTrack()", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffTrackDzError", xtitle="dzError() via PC::bestTrack()", **_commonLabelSize),
])

_packedCandidateCovarianceMatrix1 = PlotGroup("covarianceMatrix1", [
    Plot("diffCovQoverpQoverp", xtitle="cov(qoverp, qoverp)", **_commonLabelSize),
    Plot("diffCovQoverpQoverpStatus", **_commonStatus),
    Plot("diffCovQoverpQoverpUnderOverFlowSign", xtitle="cov(qoverp, qoverp)", **_common),
    Plot("diffCovLambdaLambda", xtitle="cov(lambda, lambda)", **_commonLabelSize),
    Plot("diffCovLambdaLambdaStatus", **_commonStatus),
    Plot("diffCovLambdaLambdaUnderOverFlowSign", xtitle="cov(lambda, lambda)", **_common),
    Plot("diffCovLambdaDsz", xtitle="cov(lambda, dsz)", **_commonLabelSize),
    Plot("diffCovLambdaDszStatus", **_commonStatus),
    Plot("diffCovLambdaDszUnderOverFlowSign", xtitle="cov(lambda, dsz)", **_common),
    Plot("diffCovPhiPhi", xtitle="cov(phi, phi)", **_commonLabelSize),
    Plot("diffCovPhiPhiStatus", **_commonStatus),
    Plot("diffCovPhiPhiUnderOverFlowSign", xtitle="cov(phi, phi)", **_common),
],
                                              ncols=3, legendDy=_legendDy_4rows
)
_packedCandidateCovarianceMatrix2 = PlotGroup("covarianceMatrix2", [
    Plot("diffCovPhiDxy", xtitle="cov(phi, dxy)", **_commonLabelSize),
    Plot("diffCovPhiDxyStatus", **_commonStatus),
    Plot("diffCovPhiDxyUnderOverFlowSign", xtitle="cov(phi, dxy)", **_common),
    Plot("diffCovDxyDxy", xtitle="cov(dxy, dxy)", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffCovDxyDxyStatus", **_commonStatus),
    Plot("diffCovDxyDxyUnderOverFlowSign", xtitle="cov(dxy, dxy)", **_common),
    Plot("diffCovDxyDsz", xtitle="cov(dxy, dsz)", adjustMarginRight=0.02, **_commonLabelSize),
    Plot("diffCovDxyDszStatus", **_commonStatus),
    Plot("diffCovDxyDszUnderOverFlowSign", xtitle="cov(dxy, dsz)", **_common),
    Plot("diffCovDszDsz", xtitle="cov(dsz, dsz)", adjustMarginRight=0.02, **_commonLabelSize),
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
        return algo not in _possibleTrackingNonIterationColls

class TrackingSummaryTable:
    class GeneralTracks: pass
    class GeneralTracksPt09: pass
    class HighPurity: pass
    class HighPurityPt09: pass
    class BTVLike: pass
    class AK4PFJets: pass
    class Pixel: pass

    def __init__(self, section, collection=GeneralTracks):
        self._collection = collection
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
        def _getAlgoQuality(data, algo, quality):
            for label, value in six.iteritems(data):
                (a, q) = _mapCollectionToAlgoQuality(label)
                if a == algo and q == quality:
                    return value[0] # value is (value, uncertainty) tuple
            return None
        def _getN(hname):
            h = tdirectory.Get(hname)
            if not h:
                return None
            data = plotting._th1ToOrderedDict(h)
            if self._collection == TrackingSummaryTable.GeneralTracks:
                return _getAlgoQuality(data, "ootb", "")
            elif self._collection == TrackingSummaryTable.GeneralTracksPt09:
                return _getAlgoQuality(data, "ootb", "Pt09")
            elif self._collection == TrackingSummaryTable.HighPurity:
                return _getAlgoQuality(data, "ootb", "highPurity")
            elif self._collection == TrackingSummaryTable.HighPurityPt09:
                return _getAlgoQuality(data, "ootb", "highPurityPt09")
            elif self._collection == TrackingSummaryTable.BTVLike:
                return _getAlgoQuality(data, "btvLike", "")
            elif self._collection == TrackingSummaryTable.AK4PFJets:
                return _getAlgoQuality(data, "ak4PFJets", "")
            elif self._collection == TrackingSummaryTable.Pixel:
                return _getAlgoQuality(data, "pixel", "")
            else:
                raise Exception("Collection not recognized, %s" % str(self._collection))
        def _formatOrNone(num, func):
            if num is None:
                return None
            return func(num)

        n_tps = _formatOrNone(_getN("num_simul_coll"), int)
        n_m_tps = _formatOrNone(_getN("num_assoc(simToReco)_coll"), int)

        n_tracks = _formatOrNone(_getN("num_reco_coll"), int)
        n_true = _formatOrNone(_getN("num_assoc(recoToSim)_coll"), int)
        if n_tracks is not None and n_true is not None:
            n_fake = n_tracks-n_true
        else:
            n_fake = None
        n_pileup = _formatOrNone(_getN("num_pileup_coll"), int)
        n_duplicate = _formatOrNone(_getN("num_duplicate_coll"), int)

        eff = _formatOrNone(_getN("effic_vs_coll"), lambda n: "%.4f" % n)
        eff_nopt = _formatOrNone(_getN("effic_vs_coll_allPt"), lambda n: "%.4f" % n)
        fake = _formatOrNone(_getN("fakerate_vs_coll"), lambda n: "%.4f" % n)
        duplicate = _formatOrNone(_getN("duplicatesRate_coll"), lambda n: "%.4f" % n)

        ret = [eff, n_tps, n_m_tps,
               eff_nopt, fake, duplicate,
               n_tracks, n_true, n_fake, n_pileup, n_duplicate]
        if ret.count(None) == len(ret):
            return None
        return ret

    def headers(self):
        return [
            "Efficiency",
            "Number of TrackingParticles (after cuts)",
            "Number of matched TrackingParticles",
            "Efficiency (w/o pT cut)",
            "Fake rate",
            "Duplicate rate",
            "Number of tracks",
            "Number of true tracks",
            "Number of fake tracks",
            "Number of pileup tracks",
            "Number of duplicate tracks"
        ]

# Provide a "PlotGroup" interface, but provide a html page
class TrackingSeedingLayerTable:
    def __init__(self, fileName, plots, titles, isRate, **kwargs):
        self._plots = plots
        self._titles = titles
        self._fileName = fileName
        self._format = "%.4g" if isRate else "%d"

        if len(plots) != len(titles):
            raise Exception("Number of plots (%d) has to be the same as number of titles (%d)" % (len(plots), len(titles)))

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("onlyForPileup", False)

    def onlyForPileup(self):
        """Return True if the PlotGroup is intended only for pileup samples"""
        return self._onlyForPileup

    def create(self, tdirectoryNEvents, requireAllHistograms=False):
        # [plot][histo]
        for plot in self._plots:
            plot.create(tdirectoryNEvents, requireAllHistograms)

    def draw(self, legendLabels, prefix=None, directory="", *args, **kwargs):
        # Do not make the table if it would be empty
        onlyEmptyPlots = True
        for plot in self._plots:
            if not plot.isEmpty():
                onlyEmptyPlots = False
                break
        if onlyEmptyPlots:
            return []

        haveShortLabels = False
        legendLabels = legendLabels[:]
        if max(map(len, legendLabels)) > 20:
            haveShortLabels = True
            labels_short = [str(chr(ord('A')+i)) for i in range(len(legendLabels))]
            for i, ls in enumerate(labels_short):
                legendLabels[i] = "%s: %s" % (ls, legendLabels[i])
        else:
            labels_short = legendLabels

        content = [
            '<html>',
            ' <body>',
            '  <table border="1">',
            '   <tr>',
        ]


        histos_linear = []
        histos_index = []
        labels = []
        for plot, title in zip(self._plots, self._titles):
            h_tmp = []
            l_tmp = []
            for h, l in zip(plot._histograms, labels_short):
                if h is not None:
                    h_tmp.append(len(histos_linear))
                    histos_linear.append(h)
                    l_tmp.append(l)

            if len(h_tmp) > 0:
                histos_index.append(h_tmp)
                labels.append(l_tmp)
                content.extend([
                    '    <td></td>',
                    '    <td colspan="%d">%s</td>' % (len(h_tmp), title),
                ])

        if len(histos_linear) == 0:
            return []

        content.extend([
            '   </tr>',
            '   <tr>',
        ])

        xbinlabels = plotting._mergeBinLabelsX(histos_linear)
        histos_linear = plotting._th1IncludeOnlyBins(histos_linear, xbinlabels)
        if len(histos_linear) == 0:
            return []
        (histos_linear_new, xbinlabels) = plotting._th1RemoveEmptyBins(histos_linear, xbinlabels)
        # in practice either all histograms are returned, or none, but let's add a check anyway
        if len(histos_linear_new) > 0 and len(histos_linear_new) != len(histos_linear):
            raise Exception("This should never happen. len(histos_linear_new) %d != len(histos_linear) %d" % (len(histos_linear_new), len(histos_linear)))
        histos_linear = histos_linear_new
        if len(histos_linear) == 0:
            return []

        data = [ [h.GetBinContent(i) for i in range(1, h.GetNbinsX()+1)] for h in histos_linear]
        table = html.Table(["dummy"]*len(histos_linear), xbinlabels, data, None, None, None)
        data = table.tableAsRowColumn()

        for labs in labels:
            content.append('    <td></td>')
            content.extend(['    <td>%s</td>' % lab for lab in labs])
        content.extend([
            '   </tr>',
        ])

        for irow, row in enumerate(data):
            content.extend([
                '    <tr>',
                '     <td>%s</td>' % table.rowHeaders()[irow]
            ])

            for hindices in histos_index:
                for hindex in hindices:
                    item = row[hindex]
                    formatted = self._format%item if item is not None else ""
                    content.append('     <td align="right">%s</td>' % formatted)
                content.append('    <td></td>')
            del content[-1]
            content.append('    </tr>')

        content.append('  </table>')
        if haveShortLabels:
            for lab in legendLabels:
                content.append('  %s<br/>' % lab)

        content.extend([
            ' </body>',
            '<html>'
        ])

        name = self._fileName
        if prefix is not None:
            name = prefix+name
        name += ".html"
        name = os.path.join(directory, name)

        with open(name, "w") as f:
            for line in content:
                f.write(line)
                f.write("\n")
        return [name]

_dupandfakeSeedingTable = TrackingSeedingLayerTable("dupandfakeSeeding", [p.clone() for p in _dupandfakeSeedingPlots],
                                                    ["Fake rate", "Duplicate rate", "Pileup rate"], isRate=True)
_extDistSeedingTable = TrackingSeedingLayerTable("distSeeding", [p.clone() for p in _extDistSeedingPlots],
                                                 ["All tracks", "True tracks", "Fake tracks", "Duplicate tracks"], isRate=False)

def _trackingFolders(lastDirName="Track"):
    return [
        "DQMData/Run 1/Tracking/Run summary/"+lastDirName,
        "DQMData/Tracking/"+lastDirName,
        "DQMData/Run 1/RecoTrackV/Run summary/"+lastDirName,
        "DQMData/RecoTrackV/"+lastDirName,
    ]

_simBasedPlots = [
    _effandfakePtEtaPhi,
    _effandfakeDxyDzBS,
    _effandfakeDxyDzPV,
    _effandfakeHitsLayers,
    _effandfakePos,
    _effandfakeDeltaRPU,
    _duplicateAlgo,
]
_recoBasedPlots = [
    _dupandfakePtEtaPhi,
    _dupandfakeDxyDzBS,
    _dupandfakeDxyDzPV,
    _dupandfakeHitsLayers,
    _dupandfakePos,
    _dupandfakeDeltaRPU,
    _dupandfakeChi2Seeding,
    _dupandfakeSeedingTable,
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
    _dupandfakePtEtaPhi,
    _dupandfakeDxyDzBS,
    _dupandfakeDxyDzPV,
    _dupandfakeHitsLayers,
    _dupandfakePos,
    _dupandfakeDeltaRPU,
    _dupandfakeChi2Seeding,
    _dupandfakeSeedingTable,
    _hitsAndPt,
] + _makeMVAPlots(1) \
  + _makeMVAPlots(2) \
  + _makeMVAPlots(2, hp=True) \
  + _makeMVAPlots(3) \
  + _makeMVAPlots(3, hp=True)
# add more if needed
_buildingExtendedPlots = [
    _pulls,
    _resolutionsEta,
    _resolutionsPt,
    _tuning,
]
_extendedPlots = [
    _extDistPtEtaPhi,
    _extDistDxyDzBS,
    _extDistDxyDzPV,
    _extDistHitsLayers,
    _extDistPos,
    _extDistDeltaR,
    _extDistChi2Seeding,
    _extDistSeedingTable,
    _extResidualEta,
    _extResidualPt,
    _extNrecVsNsim,
    _extHitsLayers,
    _extDistSimPtEtaPhi,
    _extDistSimDxyDzBS,
    _extDistSimDxyDzPV,
    _extDistSimHitsLayers,
    _extDistSimPos,
    _extDistSimDeltaR,
]
_summaryPlots = [
    _summary,
    _summaryN,
    _summaryByOriginalAlgo,
    _summaryByOriginalAlgoN,
    _summaryByAlgoMask,
    _summaryByAlgoMaskN,
    _summaryPt09,
    _summaryPt09N,
]
_summaryPlotsHp = [
    _summaryHp,
    _summaryNHp,
    _summaryByOriginalAlgoHp,
    _summaryByOriginalAlgoNHp,
    _summaryByAlgoMaskHp,
    _summaryByAlgoMaskNHp,
    _summaryPt09Hp,
    _summaryPt09NHp,
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
    _packedCandidateLayers,
]
plotter = Plotter()
plotterExt = Plotter()
def _appendTrackingPlots(lastDirName, name, algoPlots, onlyForPileup=False, onlyForElectron=False, onlyForConversion=False, onlyForBHadron=False, seeding=False, building=False, rawSummary=False, highPuritySummary=True):
    folders = _trackingFolders(lastDirName)
    # to keep backward compatibility, this set of plots has empty name
    limiters = dict(onlyForPileup=onlyForPileup, onlyForElectron=onlyForElectron, onlyForConversion=onlyForConversion, onlyForBHadron=onlyForBHadron)
    commonForTPF = dict(purpose=PlotPurpose.TrackingIteration, fallbackRefFiles=[
        _trackingRefFileFallbackSLHC_Phase1PU140
    ], **limiters)
    common = dict(fallbackDqmSubFolders=[
        _trackingSubFoldersFallbackSLHC_Phase1PU140,
        _trackingSubFoldersFallbackFromPV, _trackingSubFoldersFallbackConversion,
        _trackingSubFoldersFallbackPreSplitting])
    plotter.append(name, folders, TrackingPlotFolder(*algoPlots, **commonForTPF), **common)
    extendedPlots = []
    if building:
        extendedPlots.extend(_buildingExtendedPlots)
    extendedPlots.extend(_extendedPlots)
    plotterExt.append(name, folders, TrackingPlotFolder(*extendedPlots, **commonForTPF), **common)

    summaryName = ""
    if name != "":
        summaryName += name+"_"
    summaryName += "summary"
    summaryPlots = []
    if rawSummary:
        summaryPlots.extend([_summaryRaw, _summaryRawN])
    summaryPlots.extend(_summaryPlots)

    common = dict(loopSubFolders=False, purpose=PlotPurpose.TrackingSummary, page="summary",
                  #numberOfEventsHistogram=_trackingNumberOfEventsHistogram, # FIXME
                  **limiters)
    plotter.append(summaryName, folders,
                   PlotFolder(*summaryPlots, section=name, **common))
    if highPuritySummary:
        plotter.append(summaryName+"_highPurity", folders,
                       PlotFolder(*_summaryPlotsHp, section=name+"_highPurity" if name != "" else "highPurity", **common),
                       fallbackNames=[summaryName]) # backward compatibility for release validation, the HP plots used to be in the same directory with all-track plots
    if seeding:
        plotter.append(summaryName+"_seeds", folders,
                       PlotFolder(*_summaryPlotsSeeds, section=name+"_seeds", **common))

    plotter.appendTable(summaryName, folders, TrackingSummaryTable(section=name))
    plotter.appendTable(summaryName, folders, TrackingSummaryTable(section=name+"Pt09", collection=TrackingSummaryTable.GeneralTracksPt09))
    if highPuritySummary:
        sectionName = name+"_highPurity" if name != "" else "highPurity"
        plotter.appendTable(summaryName+"_highPurity", folders, TrackingSummaryTable(section=sectionName, collection=TrackingSummaryTable.HighPurity))
        plotter.appendTable(summaryName+"_highPurity", folders, TrackingSummaryTable(section=sectionName+"Pt09", collection=TrackingSummaryTable.HighPurityPt09))
    if name == "":
        plotter.appendTable(summaryName, folders, TrackingSummaryTable(section="btvLike", collection=TrackingSummaryTable.BTVLike))
        plotter.appendTable(summaryName, folders, TrackingSummaryTable(section="ak4PFJets", collection=TrackingSummaryTable.AK4PFJets))
_appendTrackingPlots("Track", "", _simBasedPlots+_recoBasedPlots)
_appendTrackingPlots("TrackTPPtLess09", "tpPtLess09", _simBasedPlots)
_appendTrackingPlots("TrackAllTPEffic", "allTPEffic", _simBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPV", "fromPV", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPVAllTP", "fromPVAllTP", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackFromPVAllTP2", "fromPVAllTP2", _simBasedPlots+_recoBasedPlots, onlyForPileup=True)
_appendTrackingPlots("TrackSeeding", "seeding", _seedingBuildingPlots, seeding=True)
_appendTrackingPlots("TrackBuilding", "building", _seedingBuildingPlots, building=True)
_appendTrackingPlots("TrackConversion", "conversion", _simBasedPlots+_recoBasedPlots, onlyForConversion=True, rawSummary=True, highPuritySummary=False)
_appendTrackingPlots("TrackGsf", "gsf", _simBasedPlots+_recoBasedPlots, onlyForElectron=True, rawSummary=True, highPuritySummary=False)
_appendTrackingPlots("TrackBHadron", "bhadron", _simBasedPlots+_recoBasedPlots, onlyForBHadron=True)
# Pixel tracks
_common = dict(purpose=PlotPurpose.Pixel, page="pixel")
plotter.append("pixelTrack", _trackingFolders("PixelTrack"), TrackingPlotFolder(*(_simBasedPlots+_recoBasedPlots), **_common))
plotterExt.append("pixelTrack", _trackingFolders("PixelTrack"), TrackingPlotFolder(*_extendedPlots, **_common))
plotter.append("pixelTrack_summary",  _trackingFolders("PixelTrack"), PlotFolder(_summaryRaw, _summaryRawN, loopSubFolders=False, purpose=PlotPurpose.TrackingSummary, page="summary", section="pixel"))
plotter.appendTable("pixelTrack_summary", _trackingFolders("PixelTrack"), TrackingSummaryTable(section="pixel", collection=TrackingSummaryTable.Pixel))


# MiniAOD
plotter.append("packedCandidate", _trackingFolders("PackedCandidate"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate"))
plotter.append("packedCandidateLostTracks", _trackingFolders("PackedCandidate/lostTracks"),
               PlotFolder(*_packedCandidatePlots, loopSubFolders=False,
                          purpose=PlotPurpose.MiniAOD, page="miniaod", section="PackedCandidate (lostTracks)"))

# HLT
_hltFolder = [
    "DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTtp",
]
plotterHLT = Plotter()
plotterHLTExt = Plotter()
_common = dict(purpose=PlotPurpose.HLT, page="hlt")
plotterHLT.append("hlt", _hltFolder, TrackingPlotFolder(*(_simBasedPlots+_recoBasedPlots), **_common))
plotterHLTExt.append("hlt", _hltFolder, TrackingPlotFolder(*_extendedPlots, **_common))

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
        # it's fine to include e.g. quadruplets here also for pair
        # steps, as non-existing modules are just ignored
        _set(seeding, "_seeding", [self._name+"SeedingLayers", self._name+"TrackingRegions", self._name+"HitDoublets", self._name+"HitTriplets", self._name+"HitQuadruplets", self._name+"Seeds"])
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
                       "initialStepTrackingRegionsPreSplitting",
                       "initialStepHitDoubletsPreSplitting",
                       "initialStepHitTripletsPreSplitting",
                       "initialStepHitQuadrupletsPreSplitting",
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
                         "initialStep",
                         "initialStepSelector"],
              building=["initialStepTrackCandidatesMkFitInput",
                        "initialStepTrackCandidatesMkFit",
                        "initialStepTrackCandidates"],
              other=["firstStepPrimaryVerticesUnsorted",
                     "initialStepTrackRefsForJets",
                     "caloTowerForTrk",
                     "ak4CaloJetsForTrk",
                     "firstStepPrimaryVertices"]),
    Iteration("highPtTripletStep",
              selection=["highPtTripletStepClassifier1",
                         "highPtTripletStepClassifier2",
                         "highPtTripletStepClassifier3",
                         "highPtTripletStep",
                         "highPtTripletStepSelector"]),
    Iteration("detachedQuadStep",
              selection=["detachedQuadStepClassifier1",
                         "detachedQuadStepClassifier2",
                         "detachedQuadStep",
                         "detachedQuadStepSelector"]),
    Iteration("detachedTripletStep",
              selection=["detachedTripletStepClassifier1",
                         "detachedTripletStepClassifier2",
                         "detachedTripletStep",
                         "detachedTripletStepSelector"]),
    Iteration("lowPtQuadStep",
              selection=["lowPtQuadStepClassifier1",
                         "lowPtQuadStepClassifier2",
                         "lowPtQuadStep",
                         "lowPtQuadStepSelector"]),
    Iteration("lowPtTripletStep",
              selection=["lowPtTripletStepClassifier1",
                         "lowPtTripletStepClassifier2",
                         "lowPtTripletStep",
                         "lowPtTripletStepSelector"]),
    Iteration("pixelPairStep",
              seeding=["pixelPairStepSeedLayers",
                       "pixelPairStepSeedLayersB",
                       "pixelPairStepSeedLayersC",
                       "pixelPairStepTrackingRegions",
                       "pixelPairStepTrackingRegionsB",
                       "pixelPairStepTrackingRegionsC",
                       "pixelPairStepTrackingRegionsSeedLayersB",
                       "pixelPairStepHitDoublets",
                       "pixelPairStepHitDoubletsB",
                       "pixelPairStepHitDoubletsC",
                       "pixelPairStepSeedsA",
                       "pixelPairStepSeedsB",
                       "pixelPairStepSeedsC",
                       "pixelPairStepSeeds",],
              selection=["pixelPairStep",
                         "pixelPairStepSelector"]),
    Iteration("mixedTripletStep",
              seeding=["mixedTripletStepSeedLayersA",
                       "mixedTripletStepSeedLayersB",
                       "mixedTripletStepTrackingRegionsA",
                       "mixedTripletStepTrackingRegionsB",
                       "mixedTripletStepHitDoubletsA",
                       "mixedTripletStepHitDoubletsB",
                       "mixedTripletStepHitTripletsA",
                       "mixedTripletStepHitTripletsB",
                       "mixedTripletStepSeedsA",
                       "mixedTripletStepSeedsB",
                       "mixedTripletStepSeeds"],
              selection=["mixedTripletStepClassifier1",
                         "mixedTripletStepClassifier2",
                         "mixedTripletStep",
                         "mixedTripletStepSelector"]),
    Iteration("pixelLessStep",
              selection=["pixelLessStepClassifier1",
                         "pixelLessStepClassifier2",
                         "pixelLessStep",
                         "pixelLessStepSelector"]),
    Iteration("tobTecStep",
              seeding=["tobTecStepSeedLayersTripl",
                       "tobTecStepSeedLayersPair",
                       "tobTecStepTrackingRegionsTripl",
                       "tobTecStepTrackingRegionsPair",
                       "tobTecStepHitDoubletsTripl",
                       "tobTecStepHitDoubletsPair",
                       "tobTecStepHitTripletsTripl",
                       "tobTecStepSeedsTripl",
                       "tobTecStepSeedsPair",
                       "tobTecStepSeeds"],
              selection=["tobTecStepClassifier1",
                         "tobTecStepClassifier2",
                         "tobTecStep",
                         "tobTecStepSelector"]),
    Iteration("jetCoreRegionalStep",
              clusterMasking=[],
              other=["jetsForCoreTracking",
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
                         "muonSeededTracksInOutSelector",
                         "muonSeededTracksOutIntClassifier",
                         "muonSeededTracksOutIntSelector"],
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
    Iteration("Other", clusterMasking=[], seeding=[], building=[], fit=[], selection=[],
              other=["trackerClusterCheckPreSplitting",
                     "trackerClusterCheck"]),
]

def _iterModuleMap(includeConvStep=True, onlyConvStep=False):
    iterations = _iterations
    if not includeConvStep:
        iterations = [i for i in iterations if i.name() != "ConvStep"]
    if onlyConvStep:
        iterations = [i for i in iterations if i.name() == "ConvStep"]
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

class TimePerEventPlot:
    def __init__(self, name, timeHisto):
        self._name = name
        self._timeHisto = timeHisto
        self._eventsHisto = "path time_real"
        self._cache = {}

    def __str__(self):
        return self._name

    def _create(self, tdirectory):
        timeTh1 = plotting._getOrCreateObject(tdirectory, self._timeHisto)
        if timeTh1 is None:
            return None

        eventsTh1 = plotting._getOrCreateObject(tdirectory, self._eventsHisto)
        if eventsTh1 is None:
            return None
        nevents = eventsTh1.GetEntries()
        if nevents == 0:
            return None

        ret = timeTh1.Clone(self._name)
        xaxis = ret.GetXaxis()
        for i in range(1, ret.GetNbinsX()+1):
            ret.SetBinContent(i, ret.GetBinContent(i)/nevents)
            ret.SetBinError(i, ret.GetBinError(i)/nevents)
            xaxis.SetBinLabel(i, xaxis.GetBinLabel(i).replace(" (unscheduled)", ""))
        return ret

    def create(self, tdirectory):
        path = tdirectory.GetPath()
        if path not in self._cache:
            self._cache[path] = self._create(tdirectory)
        return self._cache[path]

class TimePerTrackPlot:
    def __init__(self, name, timeHisto, selectedTracks=False):
        self._name = name
        self._timeHisto = timeHisto
        self._selectedTracks = selectedTracks

    def __str__(self):
        return self._name

    def _getDirectory(self, tfile):
        for dirName in _trackingFolders():
            tdir = tfile.Get(dirName)
            if tdir != None:
                return tdir
        return None

    def create(self, tdirectory):
        timeTh1 = plotting._getOrCreateObject(tdirectory, self._timeHisto)
        if timeTh1 is None:
            return None

        # this is bit of a hack, but as long as it is needed only
        # here, I won't invest in better solution
        tfile = tdirectory.GetFile()
        trkDir = self._getDirectory(tfile)
        if trkDir is None:
            return None

        iterMap = copy.copy(_collLabelMapHp)
        del iterMap["generalTracks"]
        del iterMap["jetCoreRegionalStep"] # this is expensive per track on purpose
        if self._selectedTracks:
            renameBin = lambda bl: _summaryBinRename(bl, highPurity=True, byOriginalAlgo=False, byAlgoMask=True, ptCut=False, seeds=False)
        else:
            renameBin = lambda bl: _summaryBinRename(bl, highPurity=False, byOriginalAlgo=False, byAlgoMask=False, ptCut=False, seeds=False)
        recoAB = AggregateBins("tmp", "num_reco_coll", mapping=iterMap,ignoreMissingBins=True, renameBin=renameBin)
        h_reco_per_iter = recoAB.create(trkDir)
        if h_reco_per_iter is None:
            return None
        values = {}
        for i in range(1, h_reco_per_iter.GetNbinsX()+1):
            values[h_reco_per_iter.GetXaxis().GetBinLabel(i)] = h_reco_per_iter.GetBinContent(i)


        result = []
        for i in range(1, timeTh1.GetNbinsX()+1):
            iterName = timeTh1.GetXaxis().GetBinLabel(i)
            if iterName in values:
                ntrk = values[iterName]
                result.append( (iterName,
                                timeTh1.GetBinContent(i)/ntrk if ntrk > 0 else 0,
                                timeTh1.GetBinError(i)/ntrk if ntrk > 0 else 0) )

        if len(result) == 0:
            return None

        res = ROOT.TH1F(self._name, self._name, len(result), 0, len(result))
        for i, (label, value, error) in enumerate(result):
            res.GetXaxis().SetBinLabel(i+1, label)
            res.SetBinContent(i+1, value)
            res.SetBinError(i+1, error)

        return res

class TrackingIterationOrder:
    def __init__(self):
        self._cache = {}

    def _findOrder(self, f):
        h = f.Get(_trackingIterationOrderHistogram)
        if not h:
            return None
        xaxis = h.GetXaxis()
        def _edit(s):
            # remove "Tracks" from the track producer name to get the iteration name
            # muonSeeded iterations do not have "Step" in the producer name, so add it here
            return s.replace("Tracks", "").replace("muonSeeded", "muonSeededStep")
        return [_edit(xaxis.GetBinLabel(i)) for i in range(1, h.GetNbinsX()+1)]

    def __call__(self, tdirectory, labels):
        ret = list(range(0, len(labels)))
        f = tdirectory.GetFile()
        if not f:
            return ret

        if not f.GetName() in self._cache:
            r = self._findOrder(f)
            if r is None:
                return ret
            self._cache[f.GetName()] = r
        order = self._cache[f.GetName()]

        # O(N^2) I know, but we're talking about O(10) elements...
        orderIndices = []
        for l in order:
            try:
                orderIndices.append(labels.index(l))
            except ValueError:
                pass
        ret = []
        for i, l in enumerate(labels):
            if l in order:
                try:
                    found = orderIndices.index(i)
                    if found == 0:
                        ret.append(i)
                    else:
                        ret.append(orderIndices[0])
                except ValueError:
                    ret.append(orderIndices[0])
                orderIndices.pop(0)
            else:
                ret.append(i)
        return ret

_time_per_event_cpu = TimePerEventPlot("timePerEvent", "module_time_thread_total")
_time_per_event_real = TimePerEventPlot("timePerEvent", "module_time_real_total")

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

    def _getValues(self, tdirectory, histo):
        h = tdirectory.Get(histo)
        totalReco = None
        if h:
            totalReco = "%.1f" % h.Integral()

        creator = AggregateBins("iteration", histo, _iterModuleMap(includeConvStep=False), ignoreMissingBins=True)
        h = creator.create(tdirectory)
        totalTracking = None
        if h:
            totalTracking = "%.1f" % h.Integral()

        creator = AggregateBins("iteration", histo, _iterModuleMap(onlyConvStep=True), ignoreMissingBins=True)
        h = creator.create(tdirectory)
        totalConvStep = None
        if h:
            totalConvStep = "%.1f" % h.Integral()

        return [
            totalReco,
            totalTracking,
            totalConvStep,
        ]

    def create(self, tdirectory):
        cpuValues = self._getValues(tdirectory, _time_per_event_cpu)
        realValues = self._getValues(tdirectory, _time_per_event_real)

        return cpuValues + realValues

    def headers(self):
        return [
            "Average reco CPU time / event (ms)",
            "Average tracking (w/o convStep) CPU time / event (ms)",
            "Average convStep CPU time / event (ms)",
            "Average reco real time / event (ms)",
            "Average tracking (w/o convStep) real time / event (ms)",
            "Average convStep real time / event (ms)",
        ]

_common = {
    "drawStyle": "P",
    "xbinlabelsize": 10,
    "xbinlabeloption": "d"
}

_iteration_reorder = TrackingIterationOrder()
_time_per_iter_cpu = AggregateBins("iteration", _time_per_event_cpu, _iterModuleMap(), ignoreMissingBins=True, reorder=_iteration_reorder)
_time_per_iter_real = AggregateBins("iteration", _time_per_event_real, _iterModuleMap(), ignoreMissingBins=True, reorder=_iteration_reorder)

_timing_summaryCPU = PlotGroup("summaryCPU", [
    Plot(_time_per_iter_cpu,
         ytitle="Average CPU time (ms)", title="Average CPU time / event", legendDx=-0.4, **_common),
    Plot(AggregateBins("iteration_fraction", _time_per_event_cpu, _iterModuleMap(), ignoreMissingBins=True, reorder=_iteration_reorder),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
    #
    Plot(AggregateBins("step", _time_per_event_cpu, _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Average CPU time (ms)", title="Average CPU time / event", **_common),
    Plot(AggregateBins("step_fraction", _time_per_event_cpu, _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
    #
    Plot(TimePerTrackPlot("iteration_track", _time_per_iter_cpu, selectedTracks=False),
         ytitle="Average CPU time / built track (ms)", title="Average CPU time / built track", **_common),
    Plot(TimePerTrackPlot("iteration_trackhp", _time_per_iter_cpu, selectedTracks=True),
         ytitle="Average CPU time / selected track (ms)", title="Average CPU time / selected HP track by algoMask", **_common),
    ],
)
_timing_summaryReal = PlotGroup("summaryReal", [
    Plot(_time_per_iter_real,
         ytitle="Average real time (ms)", title="Average real time / event", legendDx=-0.4, **_common),
    Plot(AggregateBins("iteration_fraction", _time_per_event_real, _iterModuleMap(), ignoreMissingBins=True, reorder=_iteration_reorder),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
    #
    Plot(AggregateBins("step", _time_per_event_real, _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Average real time (ms)", title="Average real time / event", **_common),
    Plot(AggregateBins("step_fraction", _time_per_event_real, _stepModuleMap(), ignoreMissingBins=True),
         ytitle="Fraction", title="", normalizeToUnitArea=True, **_common),
    #
    Plot(TimePerTrackPlot("iteration_track", _time_per_iter_real, selectedTracks=False),
         ytitle="Average real time / built track (ms)", title="Average real time / built track", **_common),
    Plot(TimePerTrackPlot("iteration_trackhp", _time_per_iter_real, selectedTracks=True),
         ytitle="Average real time / selected track (ms)", title="Average real time / selected HP track by algoMask", **_common),
    ],
)

_timing_iterationsCPU = PlotGroup("iterationsCPU", [
    Plot(AggregateBins(i.name(), _time_per_event_cpu, collections.OrderedDict(i.modules()), ignoreMissingBins=True),
         ytitle="Average CPU time (ms)", title=i.name(), **_common)
    for i in _iterations
],
                               ncols=4, legend=False
)
_timing_iterationsReal = PlotGroup("iterationsReal", [
    Plot(AggregateBins(i.name(), _time_per_event_real, collections.OrderedDict(i.modules()), ignoreMissingBins=True),
         ytitle="Average real time (ms)", title=i.name(), **_common)
    for i in _iterations
],
                               ncols=4, legend=False
)

# TODO: to be updated to new FastTimerService format later
#_pixelTiming = PlotGroup("pixelTiming", [
#    Plot(AggregateBins("pixel", "reconstruction_step_module_average", {"pixelTracks": ["pixelTracks"]}), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST")
#])

_timeFolders = [
#    "DQMData/Run 1/DQM/Run summary/TimerService/process RECO paths/path reconstruction_step",
    "DQMData/Run 1/DQM/Run summary/TimerService/process RECO paths/path prevalidation_step", # because of unscheduled, it's actually prevalidation_step that has all the tracking modules?
]
timePlotter = Plotter()
timePlotter.append("timing", _timeFolders, PlotFolder(
    _timing_summaryCPU,
    _timing_iterationsCPU,
    _timing_summaryReal,
    _timing_iterationsReal,
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
