from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, Plotter, AlgoOpt
import Validation.RecoTrack.plotting.validation as validation

_maxPU = 80
_maxVtx = 60
_maxEff = 1.025
_maxFake = [0.05, 0.1, 0.2, 0.5, 1.025]

_common = {"xlabel": "Pileup interactions", "xmax": _maxPU, "ymax": _maxVtx}
_recovsgen = PlotGroup("recovsgen", [
    Plot("RecoVtx_vs_GenVtx", ytitle="Reco vertices", **_common),
    Plot("MatchedRecoVtx_vs_GenVtx", ytitle="Matched reco vertices", **_common),
    Plot("merged_vs_ClosestVertexInZ", xtitle="Closest distance in Z (cm)", ytitle="Merge rate", xlog=True),
    Plot("TruePVLocationIndexCumulative", xtitle="Signal PV status in Reco collection", ytitle="Fraction of events", title="", drawStyle="hist", normalizeToUnitArea=True, xbinlabels=["Not reconstructed", "Reco and identified", "Reco, not identified"], xbinlabelsize=0.045, xbinlabeloption="h", xgrid=False, ylog=True, ymin=1e-3),
    Plot("MisTagRate_vs_PU", xtitle="PU", ytitle="Mistag rate vs. PU", xmax=_maxPU, ymax=_maxFake),
    Plot("MisTagRate_vs_sum-pt2", xtitle="#Sigmap_{T}^{2}", ytitle="Mistag rate vs. #Sigmap_{T}^{2}", xlog=True, ymax=_maxFake)
])
_effandfake = PlotGroup("effandfake", [
    Plot("effic_vs_NumVertices", xtitle="Reco vertices", ytitle="Efficiency vs. N vertices", xmax=_maxVtx, ymax=_maxEff),
    Plot("fakerate_vs_PU", xtitle="Pileup interactions", ytitle="Fake rate vs. PU", xmax=_maxPU, ymax=_maxFake),
    Plot("effic_vs_NumTracks", xtitle="Tracks", ytitle="Efficiency vs. N tracks", title="", ymax=_maxEff),
    Plot("fakerate_vs_NumTracks", xtitle="Tracks", ytitle="Fake rate vs. N tracks", title="", ymax=_maxFake),
    Plot("effic_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Efficiency vs. sum p_{T}^{2}", xlog=True, ymax=_maxEff),
    Plot("fakerate_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Fake rate vs. sum p_{T}^{2}", xlog=True, ymax=_maxFake),
])
_common = {"drawStyle": "HIST"}
_genpos = PlotGroup("genpos", [
    Plot("GenAllV_X", xtitle="Gen AllV pos x", ytitle="N", **_common),
    Plot("GenPV_X", xtitle="Gen PV pos x", ytitle="N", **_common),
    Plot("GenAllV_Y", xtitle="Gen AllV pos y", ytitle="N", **_common),
    Plot("GenPV_Y", xtitle="Gen PV pos y", ytitle="N", **_common),
    Plot("GenAllV_Z", xtitle="Gen AllV pos z", ytitle="N", **_common),
    Plot("GenPV_Z", xtitle="Gen PV pos z", ytitle="N", **_common),
])

plotter = Plotter([
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertex",
    "DQMData/Vertexing/PrimaryVertex",
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV",
    "DQMData/Vertexing/PrimaryVertexV",
],[
    _recovsgen,
    _effandfake,
])
plotterGen = Plotter([
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertex",
    "DQMData/Vertexing/PrimaryVertex",
    "DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV",
    "DQMData/Vertexing/PrimaryVertexV",
],[
     _genpos
])

class VertexValidation(validation.Validation):
    def _init__(self, *args, **kwargs):
        super(TrackingValidation, self).__init__(*args, **kwargs)

    def _getDirectoryName(self, quality, algo):
        return algo

    def _getSelectionName(self, quality, algo):
        if algo is None:
            return ""
        return "_"+algo
