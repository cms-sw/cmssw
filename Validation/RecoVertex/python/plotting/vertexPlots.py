from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter
import Validation.RecoTrack.plotting.validation as validation
from Validation.RecoTrack.plotting.html import PlotPurpose


_minPU = [0, 80, 120]
_maxPU = [80, 100, 150, 200, 250]
_minVtx = [0, 80, 120]
_maxVtx = [60, 100, 150, 200, 250]
_maxEff = 1.025
_maxFake = [0.05, 0.1, 0.2, 0.5, 0.7, 1.025]

_common = {"xtitle": "Simulated interactions", "xmin": _minPU, "xmax": _maxPU, "ymin": _minVtx, "ymax": _maxVtx}
_recovsgen = PlotGroup("recovsgen", [
    Plot("RecoVtx_vs_GenVtx", ytitle="Reco vertices", **_common),
    Plot("MatchedRecoVtx_vs_GenVtx", ytitle="Matched reco vertices", **_common),
    Plot("merged_vs_ClosestVertexInZ", xtitle="Closest distance in Z (cm)", ytitle="Merge rate", xlog=True),
],
                       legendDy=-0.025
)
_pvtagging = PlotGroup("pvtagging", [
    Plot("TruePVLocationIndexCumulative", xtitle="Signal PV status in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, xbinlabels=["Not reconstructed", "Reco and identified", "Reco, not identified"], xbinlabelsize=15, xbinlabeloption="h", xgrid=False, ylog=True, ymin=1e-3),
    Plot("TruePVLocationIndex", xtitle="Index of signal PV in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, ylog=True, ymin=1e-5),
    Plot("MisTagRate_vs_PU", xtitle="PU", ytitle="Mistag rate vs. PU", title="", xmax=_maxPU, ymax=_maxFake),
    Plot("MisTagRate_vs_sum-pt2", xtitle="#Sigmap_{T}^{2}", ytitle="Mistag rate vs. #Sigmap_{T}^{2}", title="", xlog=True, ymax=_maxFake),
],
                       legendDy=-0.025
)
_effandfake = PlotGroup("effandfake", [
    Plot("effic_vs_NumVertices", xtitle="Reco vertices", ytitle="Efficiency vs. N vertices", xmin=_minVtx, xmax=_maxVtx, ymax=_maxEff),
    Plot("fakerate_vs_PU", xtitle="Pileup interactions", ytitle="Fake rate vs. PU", xmin=_minPU, xmax=_maxPU, ymax=_maxFake),
    Plot("effic_vs_NumTracks", xtitle="Tracks", ytitle="Efficiency vs. N tracks", title="", ymax=_maxEff),
    Plot("fakerate_vs_NumTracks", xtitle="Tracks", ytitle="Fake rate vs. N tracks", title="", ymax=_maxFake),
    Plot("effic_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Efficiency vs. sum p_{T}^{2}", xlog=True, ymax=_maxEff),
    Plot("fakerate_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Fake rate vs. sum p_{T}^{2}", xlog=True, ymax=_maxFake),
])
_common = {"title": "", "xtitle": "Number of tracks", "scale": 1e4, "ylog": True, "ymin": 5, "ymax": 500}
_resolution = PlotGroup("resolution", [
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_NumTracks_Sigma", ytitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_NumTracks_Sigma", ytitle="Resolution in x for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_NumTracks_Sigma", ytitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_NumTracks_Sigma", ytitle="Resolution in y for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_NumTracks_Sigma", ytitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_NumTracks_Sigma", ytitle="Resolution in z for merged vertices (#mum)", **_common),
])
_common = {"stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "xmin": -6, "xmax": 6, "ylog": True, "ymin": 5e-5, "ymax": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]}
_pull = PlotGroup("pull", [
    Plot("RecoAllAssoc2GenMatched_PullX", xtitle="x", ytitle="Pull vs. x", **_common),
    Plot("RecoAllAssoc2GenMatched_PullY", xtitle="y", ytitle="Pull vs. y", **_common),
    Plot("RecoAllAssoc2GenMatched_PullZ", xtitle="z", ytitle="Pull vs. z", **_common),
],
                  legendDy=-0.025
)
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
    Plot("RecoAssoc2GenPVMatched_Pt2", xtitle="#sum^{}p_{T}^{2}", ytitle="Reco vertices matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVMatchedNotHighest_Pt2", xtitle="#sum^{}p_{T}^{2}", ytitle="Reco non-PV-vertices matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVNotMatched_Pt2", xtitle="#sum^{}p_{T}^{2}", ytitle="Reco vertices not matched to gen PV", **_common),
    Plot("RecoAssoc2GenPVNotMatched_GenPVTracksRemoved_Pt2", xtitle="#sum^{}p_{T}^{2}, gen PV tracks removed", ytitle="Reco vertices not matched to gen PV", **_common),
],
                    legendDy=-0.025
)

_common = {"drawStyle": "HIST"}
_genpos = PlotGroup("genpos", [
    Plot("GenAllV_X", xtitle="Gen AllV pos x", ytitle="N", **_common),
    Plot("GenPV_X", xtitle="Gen PV pos x", ytitle="N", **_common),
    Plot("GenAllV_Y", xtitle="Gen AllV pos y", ytitle="N", **_common),
    Plot("GenPV_Y", xtitle="Gen PV pos y", ytitle="N", **_common),
    Plot("GenAllV_Z", xtitle="Gen AllV pos z", ytitle="N", **_common),
    Plot("GenPV_Z", xtitle="Gen PV pos z", ytitle="N", **_common),
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
            for i in xrange(1, h.GetNbinsX()+1):
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
plotter = Plotter()
plotter.append("", _vertexFolders, PlotFolder(
    _recovsgen,
    _pvtagging,
    _effandfake,
    _resolution,
    _pull,
    _puritymissing,
    _sumpt2,
    purpose=PlotPurpose.Vertexing,
    page="vertex"
))
plotter.appendTable("", _vertexFolders, VertexSummaryTable())
#plotter.append("gen", _vertexFolders, PlotFolder(_genpos, loopSubFolders=False, purpose=PlotPurpose.Vertexing, page="vertex", section="Gen vertex"))

class VertexValidation(validation.Validation):
    def _init__(self, *args, **kwargs):
        super(TrackingValidation, self).__init__(*args, **kwargs)

    def _getDirectoryName(self, quality, algo):
        return algo

    def _getSelectionName(self, quality, algo):
        if algo is None:
            return ""
        return "_"+algo
