from builtins import range
import copy

from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter
import Validation.RecoTrack.plotting.validation as validation
from Validation.RecoTrack.plotting.html import PlotPurpose
from Validation.RecoTrack.plotting.trackingPlots import _legendDy_1row, _legendDy_2rows, _legendDy_2rows_3cols, _legendDy_4rows, _minMaxResidual

_minPU = [0, 10, 20, 40, 80, 120]
_maxPU = [60, 80, 100, 150, 200, 250]
_minVtx = [0, 80, 120]
_maxVtx = [60, 100, 150, 200, 250]
_maxEff = 1.025
_maxFake = [0.05, 0.1, 0.2, 0.5, 0.7, 1.025]
_minMaxRes = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000]
(_minResidual, _maxResidual) = _minMaxResidual(_minMaxRes)
_minMaxPt = [5e-1, 1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
_minPull = [0, 0.5, 0.8, 0.9]
_maxPull = [1.1, 1.2, 1.5, 2]
_minVertexZ = list(range(-60,-10,10))
_maxVertexZ = list(range(20,70,10))

_vertexNumberOfEventsHistogram = "DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV/GenPV_Z"

_common = {"xtitle": "Simulated interactions", "xmin": _minPU, "xmax": _maxPU, "ymin": _minVtx, "ymax": _maxVtx}
_recovsgen = PlotGroup("recovsgen", [
    Plot("RecoVtx_vs_GenVtx", ytitle="Reco vertices", **_common),
    Plot("MatchedRecoVtx_vs_GenVtx", ytitle="Matched reco vertices", **_common),
    Plot("merged_vs_ClosestVertexInZ", xtitle="Closest distance in Z (cm)", ytitle="Merge rate", xlog=True, xmin=1e-3, ymax=_maxFake),
    Plot("merged_vs_Z", xtitle="Z (cm)", ytitle="Merge rate", xmin=-20, xmax=20, ymax=_maxFake),
],
                       legendDy=_legendDy_2rows, onlyForPileup=True,
)
_pvtagging = PlotGroup("pvtagging", [
    Plot("TruePVLocationIndexCumulative", xtitle="Signal PV status in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, xbinlabels=["Not reconstructed", "Reco and identified", "Reco, not identified"], xbinlabelsize=15, xbinlabeloption="h", xgrid=False, ylog=True, ymin=1e-3, ratioCoverageXrange=[-0.5, 0.5]),
    Plot("TruePVLocationIndex", xtitle="Index of signal PV in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, ylog=True, ymin=1e-5),
    Plot("MisTagRate_vs_PU", xtitle="Number of simulated interactions", ytitle="Mistag rate", title="", xmax=_maxPU, ymax=_maxFake),
    Plot("MisTagRate_vs_sum-pt2", xtitle="#Sigmap_{T}^{2} (GeV^{2})", ytitle="Mistag rate", title="", xlog=True, ymax=_maxFake),
],
                       legendDy=_legendDy_2rows
)
_effandfake = PlotGroup("effandfake", [
    Plot("effic_vs_NumVertices", xtitle="Number of simulated interactions", ytitle="Efficiency", xmin=_minPU, xmax=_maxPU, ymax=_maxEff),
    Plot("fakerate_vs_PU", xtitle="Number of simulated interactions", ytitle="Fake rate", xmin=_minPU, xmax=_maxPU, ymax=_maxFake),
    Plot("effic_vs_NumTracks", xtitle="Tracks", ytitle="Efficiency", title="", ymax=_maxEff),
    Plot("fakerate_vs_NumTracks", xtitle="Tracks", ytitle="Fake rate", title="", ymax=_maxFake),
    Plot("effic_vs_Pt2", xtitle="#sum^{}p_{T}^{2}", ytitle="Efficiency", xlog=True, ymax=_maxEff),
    Plot("fakerate_vs_Pt2", xtitle="#sum^{}p_{T}^{2}", ytitle="Fake rate", xlog=True, ymax=_maxFake),
])
_common = {"title": "", "stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "ylog": True, "ymin": [5e-7, 5e-6, 5e-5, 5e-4]}
_resolution = PlotGroup("resolution", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX", xtitle="Resolution in x (#mum) for PV", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolX", xtitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX", xtitle="Resolution in x for merged vertices (#mum)", **_common),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY", xtitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY", xtitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY", xtitle="Resolution in y for merged vertices (#mum)", **_common),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ", xtitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ", xtitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ", xtitle="Resolution in z for merged vertices (#mum)", **_common),
], ncols=3)
_commonNumTracks = dict(title="", xtitle="Number of tracks", scale=1e4, ylog=True, ymin=_minMaxRes , ymax=_minMaxRes)
_resolutionNumTracks = PlotGroup("resolutionNumTracks", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_NumTracks_Sigma", ytitle="#sigma(#delta x) (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_NumTracks_Sigma", ytitle="#sigma(#delta x) (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_NumTracks_Sigma", ytitle="#sigma(#delta x) x for merged vertices (#mum)", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_NumTracks_Sigma", ytitle="#sigma(#delta y) (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_NumTracks_Sigma", ytitle="#sigma(#delta y) (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_NumTracks_Sigma", ytitle="#sigma(#delta y) for merged vertices (#mum)", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_NumTracks_Sigma", ytitle="#sigma(#delta z) (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_NumTracks_Sigma", ytitle="#sigma(#delta z) (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_NumTracks_Sigma", ytitle="#sigma(#delta z) for merged vertices (#mum)", **_commonNumTracks),
], ncols=3)
_commonPt = copy.copy(_commonNumTracks)
_commonPt.update(dict(xtitle= "#sum^{}p_{T} (GeV)", xlog=True, xmin=_minMaxPt, xmax=_minMaxPt))
_resolutionPt = PlotGroup("resolutionPt", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_Pt_Sigma", ytitle="#sigma(#delta x) (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Pt_Sigma", ytitle="#sigma(#delta x) (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Pt_Sigma", ytitle="#sigma(#delta x) for merged vertices (#mum)", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_Pt_Sigma", ytitle="#sigma(#delta y) (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Pt_Sigma", ytitle="#sigma(#delta y) (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Pt_Sigma", ytitle="#sigma(#delta y) for merged vertices (#mum)", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_Pt_Sigma", ytitle="#sigma(#delta z) (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Pt_Sigma", ytitle="#sigma(#delta z) (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Pt_Sigma", ytitle="#sigma(#delta z) for merged vertices (#mum)", **_commonPt),
], ncols=3)
_common = {"stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "xmin": -6, "xmax": 6, "ylog": True, "ymin": 5e-5, "ymax": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]}
_pull = PlotGroup("pull", [
    Plot("RecoPVAssoc2GenPVMatched_PullX", xtitle="x", ytitle="Pull of x for PV", **_common),
    Plot("RecoAllAssoc2GenMatched_PullX", xtitle="x", ytitle="Pull of x", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_PullX", xtitle="x", ytitle="Pull of x  for merged vertices", **_common),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullY", xtitle="y", ytitle="Pull of y for PV", **_common),
    Plot("RecoAllAssoc2GenMatched_PullY", xtitle="y", ytitle="Pull of y", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_PullY", xtitle="y", ytitle="Pull of y for merged vertices", **_common),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullZ", xtitle="z", ytitle="Pull of z for PV", **_common),
    Plot("RecoAllAssoc2GenMatched_PullZ", xtitle="z", ytitle="Pull of z", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_PullZ", xtitle="z", ytitle="Pull of z for merged vertices", **_common),
], ncols=3)
_commonNumTracks.update(dict(scale=1, ylog=False, ymin=_minPull, ymax=_maxPull))
_pullNumTracks = PlotGroup("pullNumTracks", [
    Plot("RecoPVAssoc2GenPVMatched_PullX_vs_NumTracks_Sigma", ytitle="Pull of x for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_PullX_vs_NumTracks_Sigma", ytitle="Pull of x", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_PullX_vs_NumTracks_Sigma", ytitle="Pull of x for merged vertices", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullY_vs_NumTracks_Sigma", ytitle="Pull of y for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_PullY_vs_NumTracks_Sigma", ytitle="Pull of y", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_PullY_vs_NumTracks_Sigma", ytitle="Pull of y for merged vertices", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullZ_vs_NumTracks_Sigma", ytitle="Pull of z for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_PullZ_vs_NumTracks_Sigma", ytitle="Pull of z", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_PullZ_vs_NumTracks_Sigma", ytitle="Pull of z for merged vertices", **_commonNumTracks),
], ncols=3)
_commonPt.update(dict(scale=1, ylog=False, ymin=_minPull, ymax=_maxPull))
_pullPt = PlotGroup("pullPt", [
    Plot("RecoPVAssoc2GenPVMatched_PullX_vs_Pt_Sigma", ytitle="Pull of x for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_PullX_vs_Pt_Sigma", ytitle="Pull of x", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_PullX_vs_Pt_Sigma", ytitle="Pull of x for merged vertices", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullY_vs_Pt_Sigma", ytitle="Pull of y for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_PullY_vs_Pt_Sigma", ytitle="Pull of y", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_PullY_vs_Pt_Sigma", ytitle="Pull of y for merged vertices", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullZ_vs_Pt_Sigma", ytitle="Pull of z for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_PullZ_vs_Pt_Sigma", ytitle="Pull of z", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_PullZ_vs_Pt_Sigma", ytitle="Pull of z for merged vertices", **_commonPt),
], ncols=3)

_common={"drawStyle": "HIST", "normalizeToUnitArea": True}
_puritymissing = PlotGroup("puritymissing", [
    Plot("RecoPVAssoc2GenPVMatched_Purity", xtitle="Purity", ytitle="Number of reco PVs matched to gen PVs", ylog=True, ymin=1e-4, **_common),
    Plot("RecoPVAssoc2GenPVNotMatched_Purity", xtitle="Purity", ytitle="Number of reco PVs not matcched to gen PVs", ylog=True, ymin=1e-3, **_common),
    Plot("RecoPVAssoc2GenPVMatched_Missing", xtitle="Fraction of reco p_{T} associated to gen PV \"missing\" from reco PV", ytitle="Number of reco PVs matched to gen PVs", ylog=True, ymin=1e-4, **_common),
    Plot("RecoPVAssoc2GenPVNotMatched_Missing", xtitle="Fraction of reco p_{T} associated to gen PV \"missing\" from reco PV", ytitle="Number of reco PVs not matcched to gen PVs", ylog=True, ymin=1e-3, **_common),
#    Plot("fakerate_vs_Purity", xtitle="Purity", ytitle="Fake rate", ymax=_maxFake),
])
# "xgrid": False, "ygrid": False,
_common={"drawStyle": "HIST", "xlog": True, "ylog": True, "ymin": 0.5}
_sumpt2 = PlotGroup("sumpt2", [
    Plot("RecoAssoc2GenPVMatched_Pt2", xtitle="#sum^{}p_{T}^{2} (GeV^{2})", ytitle="Reco vertices matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVMatchedNotHighest_Pt2", xtitle="#sum^{}p_{T}^{2} (GeV^{2})", ytitle="Reco non-PV-vertices matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVNotMatched_Pt2", xtitle="#sum^{}p_{T}^{2} (GeV^{2})", ytitle="Reco vertices not matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVNotMatched_GenPVTracksRemoved_Pt2", xtitle="#sum^{}p_{T}^{2} (GeV^{2}), gen PV tracks removed", ytitle="Reco vertices not matched to gen PV", **_common),
],
                    legendDy=_legendDy_2rows, onlyForPileup=True,
)

_k0_effandfake = PlotGroup("effandfake", [
    Plot("K0sEffVsPt", xtitle="p_{T} (GeV)", ytitle="Efficiency vs. p_{T}"),
    Plot("K0sFakeVsPt", xtitle="p_{T} (GeV)", ytitle="Fake rate vs. p_{T}"),
    Plot("K0sEffVsEta", xtitle="#eta", ytitle="Efficiency vs. #eta"),
    Plot("K0sFakeVsEta", xtitle="#eta", ytitle="Fake rate vs. #eta"),
    Plot("K0sEffVsR", xtitle="R (cm)", ytitle="Efficiency vs. R"),
    Plot("K0sFakeVsR", xtitle="R (cm)", ytitle="Fake rate vs. R"),
])
_k0_effandfakeTk = PlotGroup("effandfakeTk", [
#    Plot("K0sTkEffVsPt"),
    Plot("K0sTkFakeVsPt", xtitle="p_{T} (GeV)", ytitle="Fake rate vs. p_{T}"),
#    Plot("K0sTkEffVsEta"),
    Plot("K0sTkFakeVsEta", xtitle="#eta", ytitle="Fake rate vs. #eta"),
#    Plot("K0sTkEffVsR"),
    Plot("K0sTkFakeVsR", xtitle="R (cm)", ytitle="Fake rate vs. R"),
],
                             legendDy=_legendDy_2rows
)
_common = dict(normalizeToUnitArea=True, drawStyle="HIST", stat=True)
_k0_mass = PlotGroup("mass", [
    Plot("ksMassAll", xtitle="mass of all (GeV)", **_common),
    Plot("ksMassGood", xtitle="mass of good (GeV)", **_common),
    Plot("ksMassFake", xtitle="mass of fake (GeV)", **_common),
],
                     legendDy=_legendDy_2rows
)
_lambda_effandfake = PlotGroup("effandfake", [
    Plot("LamEffVsPt", xtitle="p_{T} (GeV)", ytitle="Efficiency vs. p_{T}"),
    Plot("LamFakeVsPt", xtitle="p_{T} (GeV)", ytitle="Fake rate vs. p_{T}"),
    Plot("LamEffVsEta", xtitle="#eta", ytitle="Efficiency vs. #eta"),
    Plot("LamFakeVsEta", xtitle="#eta", ytitle="Fake rate vs. #eta"),
    Plot("LamEffVsR", xtitle="R (cm)", ytitle="Efficiency vs. R"),
    Plot("LamFakeVsR", xtitle="R (cm)", ytitle="Fake rate vs. R"),
])
_lambda_effandfakeTk = PlotGroup("effandfakeTk", [
#    Plot("LamTkEffVsPt"),
    Plot("LamTkFakeVsPt", xtitle="p_{T} (GeV)", ytitle="Fake rate vs. p_{T}"),
#    Plot("LamTkEffVsEta"),
    Plot("LamTkFakeVsEta", xtitle="#eta", ytitle="Fake rate vs. #eta"),
#    Plot("LamTkEffVsR"),
    Plot("LamTkFakeVsR", xtitle="R (cm)", ytitle="Fake rate vs. R"),
],
                                 legendDy=_legendDy_2rows
)
_lambda_mass = PlotGroup("mass", [
    Plot("lamMassAll", xtitle="mass of all (GeV)", **_common),
    Plot("lamMassGood", xtitle="mass of good (GeV)", **_common),
    Plot("lamMassFake", xtitle="mass of fake (GeV)", **_common),
],
                         legendDy=_legendDy_2rows
)

## Extended set of plots
_common = dict(drawStyle = "HIST", stat=True)
_commonXY = dict(xmin=[x*0.1 for x in range(-6, 6, 1)], xmax=[x*0.1 for x in range(-5, 7, 1)])
_commonZ = dict(xmin=[-60,-30], xmax=[30,60])
_commonXY.update(_common)
_commonZ.update(_common)
_extGenpos = PlotGroup("genpos", [
    Plot("GenAllV_X", xtitle="Gen AllV pos x (cm)", ytitle="N", **_commonXY),
    Plot("GenPV_X",   xtitle="Gen PV pos x (cm)",   ytitle="N", **_commonXY),
    Plot("GenAllV_Y", xtitle="Gen AllV pos y (cm)", ytitle="N", **_commonXY),
    Plot("GenPV_Y",   xtitle="Gen PV pos y (cm)",   ytitle="N", **_commonXY),
    Plot("GenAllV_Z", xtitle="Gen AllV pos z (cm)", ytitle="N", **_commonZ),
    Plot("GenPV_Z",   xtitle="Gen PV pos z (cm)",   ytitle="N", **_commonZ),
])
_extDist = PlotGroup("dist", [
    Plot("RecoAllAssoc2Gen_X", xtitle="Reco vertex pos x (cm)", ytitle="N", **_commonXY),
    Plot("RecoAllAssoc2Gen_Y", xtitle="Reco vertex pos y (cm)", ytitle="N", **_commonXY),
    Plot("RecoAllAssoc2Gen_R", xtitle="Reco vertex pos r (cm)", ytitle="N", **_commonXY),
    Plot("RecoAllAssoc2Gen_Z", xtitle="Reco vertex pos z (cm)", ytitle="N", **_commonZ),
    Plot("RecoAllAssoc2Gen_NumVertices", xtitle="Number of reco vertices", ytitle="A.u.", normalizeToUnitArea=True, stat=True, drawStyle="hist", min=_minVtx, xmax=_maxVtx),
    Plot("RecoAllAssoc2Gen_NumTracks", xtitle="Number of tracks in vertex fit", ytitle="N", stat=True, drawStyle="hist"),
])
_commonZ = dict(title="", xtitle="Vertex z (cm)", scale=1e4, ylog=True, ymin=_minMaxRes , ymax=_minMaxRes, xmin=_minVertexZ, xmax=_maxVertexZ)
_extResolutionZ = PlotGroup("resolutionZ", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_Z_Sigma", ytitle="#sigma(#delta x) (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Z_Sigma", ytitle="#sigma(#delta x) (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Z_Sigma", ytitle="#sigma(#delta x) for merged vertices (#mum)", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_Z_Sigma", ytitle="#sigma(#delta y) (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Z_Sigma", ytitle="#sigma(#delta y) (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Z_Sigma", ytitle="#sigma(#delta y) for merged vertices (#mum)", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_Z_Sigma", ytitle="#sigma(#delta z) (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Z_Sigma", ytitle="#sigma(#delta z) (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Z_Sigma", ytitle="#sigma(#delta z) for merged vertices (#mum)", **_commonZ),
], ncols=3)
_commonPU = copy.copy(_commonZ)
_commonPU.update(dict(xtitle="Simulated interactions", xmin=_minPU, xmax=_maxPU))
_extResolutionPU = PlotGroup("resolutionPU", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_PU_Sigma", ytitle="Resolution in x (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_PU_Sigma", ytitle="Resolution in x (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_PU_Sigma", ytitle="Resolution in x for merged vertices (#mum)", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_PU_Sigma", ytitle="Resolution in y (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_PU_Sigma", ytitle="Resolution in y (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_PU_Sigma", ytitle="Resolution in y for merged vertices (#mum)", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_PU_Sigma", ytitle="Resolution in z (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_PU_Sigma", ytitle="Resolution in z (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_PU_Sigma", ytitle="Resolution in z for merged vertices (#mum)", **_commonPU),
], ncols=3)
_commonZ.update(dict(scale=1, ylog=False, ymin=_minPull, ymax=_maxPull))
_extPullZ = PlotGroup("pullZ", [
    Plot("RecoPVAssoc2GenPVMatched_PullX_vs_Z_Sigma", ytitle="Pull of x for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_PullX_vs_Z_Sigma", ytitle="Pull of x", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_PullX_vs_Z_Sigma", ytitle="Pull of x for merged vertices", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullY_vs_Z_Sigma", ytitle="Pull of y for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_PullY_vs_Z_Sigma", ytitle="Pull of y", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_PullY_vs_Z_Sigma", ytitle="Pull of y for merged vertices", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullZ_vs_Z_Sigma", ytitle="Pull of z for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_PullZ_vs_Z_Sigma", ytitle="Pull of z", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_PullZ_vs_Z_Sigma", ytitle="Pull of z for merged vertices", **_commonZ),
], ncols=3)
_commonPU.update(dict(scale=1, ylog=False, ymin=_minPull, ymax=_maxPull))
_extPullPU = PlotGroup("pullPU", [
    Plot("RecoPVAssoc2GenPVMatched_PullX_vs_PU_Sigma", ytitle="Pull of x for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_PullX_vs_PU_Sigma", ytitle="Pull of x", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_PullX_vs_PU_Sigma", ytitle="Pull of x for merged vertices", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullY_vs_PU_Sigma", ytitle="Pull of y for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_PullY_vs_PU_Sigma", ytitle="Pull of y", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_PullY_vs_PU_Sigma", ytitle="Pull of y for merged vertices", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_PullZ_vs_PU_Sigma", ytitle="Pull of z for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_PullZ_vs_PU_Sigma", ytitle="Pull of z", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_PullZ_vs_PU_Sigma", ytitle="Pull of z for merged vertices", **_commonPU),
], ncols=3)
_commonNumTracks.update(dict(scale=1e4, ymin=_minResidual, ymax=_maxResidual))
_extResidualNumTracks = PlotGroup("residualNumTracks", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_NumTracks_Mean", ytitle="< #delta x > (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_NumTracks_Mean", ytitle="< #delta x > (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_NumTracks_Mean", ytitle="< #delta x > for merged vertices (#mum)", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_NumTracks_Mean", ytitle="< #delta y > (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_NumTracks_Mean", ytitle="< #delta y > (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_NumTracks_Mean", ytitle="< #delta y > for merged vertices (#mum)", **_commonNumTracks),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_NumTracks_Mean", ytitle="< #delta z > (#mum) for PV", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_NumTracks_Mean", ytitle="< #delta  z > (#mum)", **_commonNumTracks),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_NumTracks_Mean", ytitle="< #delta z > for merged vertices (#mum)", **_commonNumTracks),
], ncols=3)
_commonPt.update(dict(scale=1e4, ymin=_minResidual, ymax=_maxResidual))
_extResidualPt = PlotGroup("residualPt", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_Pt_Mean", ytitle="< #delta x > (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Pt_Mean", ytitle="< #delta x > (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Pt_Mean", ytitle="< #delta x > for merged vertices (#mum)", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_Pt_Mean", ytitle="< #delta y > (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Pt_Mean", ytitle="< #delta y > (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Pt_Mean", ytitle="< #delta y > for merged vertices (#mum)", **_commonPt),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_Pt_Mean", ytitle="< #delta z > (#mum) for PV", **_commonPt),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Pt_Mean", ytitle="< #delta z > (#mum)", **_commonPt),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Pt_Mean", ytitle="< #delta z > for merged vertices (#mum)", **_commonPt),
], ncols=3)
_commonZ.update(dict(scale=1e4, ymin=_minResidual, ymax=_maxResidual))
_extResidualZ = PlotGroup("residualZ", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_Z_Mean", ytitle="< #delta x > (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Z_Mean", ytitle="< #delta x > (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Z_Mean", ytitle="< #delta x > for merged vertices (#mum)", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_Z_Mean", ytitle="< #delta y > (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Z_Mean", ytitle="< #delta y > (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Z_Mean", ytitle="< #delta y > for merged vertices (#mum)", **_commonZ),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_Z_Mean", ytitle="< #delta z > (#mum) for PV", **_commonZ),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Z_Mean", ytitle="< #delta z > (#mum)", **_commonZ),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Z_Mean", ytitle="< #delta z > for merged vertices (#mum)", **_commonZ),
], ncols=3)
_commonPU.update(dict(scale=1e4, ymin=_minResidual, ymax=_maxResidual))
_extResidualPU = PlotGroup("residualPU", [
    Plot("RecoPVAssoc2GenPVMatched_ResolX_vs_PU_Mean", ytitle="< #delta x > (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_PU_Mean", ytitle="< #delta x > (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_PU_Mean", ytitle="< #delta x > for merged vertices (#mum)", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolY_vs_PU_Mean", ytitle="< #delta y > (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_PU_Mean", ytitle="< #delta y > (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_PU_Mean", ytitle="< #delta y > for merged vertices (#mum)", **_commonPU),
    #
    Plot("RecoPVAssoc2GenPVMatched_ResolZ_vs_PU_Mean", ytitle="< #delta z > (#mum) for PV", **_commonPU),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_PU_Mean", ytitle="< #delta z > (#mum)", **_commonPU),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_PU_Mean", ytitle="< #delta z > for merged vertices (#mum)", **_commonPU),
], ncols=3)
_extDqm = PlotGroup("dqm", [
    Plot("tagVtxTrksVsZ", xtitle="z_{vertex} - z_{beamspot} (cm)", ytitle="Tracks / selected PV"),
    Plot("otherVtxTrksVsZ", xtitle="z_{vertex} - z_{beamspot} (cm)", ytitle="Tracks / pileup vertex"),
    Plot("vtxNbr", xtitle="Reconstructed vertices", ytitle="Events", stat=True, drawStyle="hist", xmin=_minVtx, xmax=_maxVtx),
])
_common = dict(ytitle="Vertices", stat=True)
_extDqmDiff = PlotGroup("dqmDiff", [
    Plot("tagDiffX", xtitle="PV x_{vertex} - x_{beamspot} (#mum)", **_common),
    Plot("otherDiffX", xtitle="Pileup vertex x_{vertex} - x_{beamspot} (#mum)", **_common),
    #
    Plot("tagDiffY", xtitle="PV y_{vertex} - y_{beamspot} (#mum)", **_common),
    Plot("otherDiffY", xtitle="Pileup vertex y_{vertex} - y_{beamspot} (#mum)", **_common),
])
_extDqmErr = PlotGroup("dqmErr", [
    Plot("tagErrX", xtitle="PV uncertainty in x (um)", **_common),
    Plot("otherErrX", xtitle="Pileup vertex uncertainty in x (um)", **_common),
    #
    Plot("otherErrY", xtitle="Pileup vertex uncertainty in y (um)", **_common),
    Plot("tagErrY", xtitle="PV uncertainty in y (um)", **_common),
    #
    Plot("otherErrZ", xtitle="Pileup vertex uncertainty in z (um)", **_common),
    Plot("tagErrZ", xtitle="PV uncertainty in z (um)", **_common),
])

class VertexSummaryTable:
    def __init__(self, page="vertex"):
        self._purpose = PlotPurpose.Vertexing
        self._page = page

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self, dqmSubFolder):
        return dqmSubFolder

    def create(self, tdirectory):
        def _formatOrNone(num, func):
            if num is None:
                return None
            return func(num)

        ret = []
        h = tdirectory.Get("TruePVLocationIndexCumulative")
        if h:
            n_events = h.GetEntries()
            n_pvtagged = h.GetBinContent(2)
            ret.extend([int(n_events), "%.4f"%(float(n_pvtagged)/float(n_events))])
        else:
            ret.extend([None, None])

        h = tdirectory.Get("globalEfficiencies")
        if h:
            d = {}
            for i in range(1, h.GetNbinsX()+1):
                d[h.GetXaxis().GetBinLabel(i)] = h.GetBinContent(i)
            ret.extend([
                _formatOrNone(d.get("effic_vs_Z", None), lambda n: "%.4f"%n),
                _formatOrNone(d.get("fakerate_vs_Z", None), lambda n: "%.4f"%n),
                _formatOrNone(d.get("merged_vs_Z", None), lambda n: "%.4f"%n),
                _formatOrNone(d.get("duplicate_vs_Z", None), lambda n: "%.4f"%n),
            ])
        else:
            ret.extend([None]*4)

        if ret.count(None) == len(ret):
            return None

        return ret

    def headers(self):
        return [
            "Events",
            "PV reco+tag efficiency",
            "Efficiency",
            "Fake rate",
            "Merge rate",
            "Duplicate rate",
        ]

_vertexFolders = [
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertex",
    "DQMData/Vertexing/PrimaryVertex",
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV",
    "DQMData/Vertexing/PrimaryVertexV",
]
_vertexDqmFolders = [
    "DQMData/Run 1/OfflinePV/Run summary/offlinePrimaryVertices",
    "DQMData/OffinePV/offlinePrimaryVertices",
]
_v0Folders = [
    "DQMData/Run 1/Vertexing/Run summary/V0",
    "DQMData/Vertexing/V0",
    "DQMData/Run 1/Vertexing/Run summary/V0V",
    "DQMData/Vertexing/V0V",
]
plotter = Plotter()
plotterExt = Plotter()
plotter.append("", _vertexFolders, PlotFolder(
    _recovsgen,
    _pvtagging,
    _effandfake,
    _resolution,
    _resolutionNumTracks,
    _resolutionPt,
    _pull,
    _pullNumTracks,
    _pullPt,
    _puritymissing,
    _sumpt2,
    purpose=PlotPurpose.Vertexing,
    page="vertex"
))
plotter.appendTable("", _vertexFolders, VertexSummaryTable())
plotter.append("K0", [x+"/K0" for x in _v0Folders], PlotFolder(
    _k0_effandfake,
    _k0_effandfakeTk,
    _k0_mass,
    loopSubFolders=False,
    purpose=PlotPurpose.Vertexing,
    page="v0", section="k0"
))
plotter.append("Lambda", [x+"/Lambda" for x in _v0Folders], PlotFolder(
    _lambda_effandfake,
    _lambda_effandfakeTk,
    _lambda_mass,
    loopSubFolders=False,
    purpose=PlotPurpose.Vertexing,
    page="v0", section="lambda"
))
plotterExt.append("", _vertexFolders, PlotFolder(
    _extDist,
    _extResolutionZ,
    _extResolutionPU,
    _extPullZ,
    _extPullPU,
    _extResidualNumTracks,
    _extResidualPt,
    _extResidualZ,
    _extResidualPU,
    purpose=PlotPurpose.Vertexing,
    page="vertex",
    onlyForPileup=True,
    numberOfEventsHistogram=_vertexNumberOfEventsHistogram
))
plotterExt.append("dqm", _vertexDqmFolders, PlotFolder(
    _extDqm,
    _extDqmDiff,
    _extDqmErr,
    loopSubFolders=False,
    purpose=PlotPurpose.Vertexing,
    page="vertex",
    section="offlinePrimaryVertices",
    onlyForPileup=True
))
plotterExt.append("gen", _vertexFolders, PlotFolder(
    _extGenpos,
    loopSubFolders=False,
    purpose=PlotPurpose.Vertexing,
    page="vertex",
    section="genvertex",
    onlyForPileup=True
))

class VertexValidation(validation.Validation):
    def _init__(self, *args, **kwargs):
        super(TrackingValidation, self).__init__(*args, **kwargs)

    def _getDirectoryName(self, quality, algo):
        return algo

    def _getSelectionName(self, quality, algo):
        if algo is None:
            return ""
        return "_"+algo
