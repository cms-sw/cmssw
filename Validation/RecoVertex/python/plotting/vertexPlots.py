from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter
import Validation.RecoTrack.plotting.validation as validation
from Validation.RecoTrack.plotting.html import PlotPurpose


_minPU = [0, 10, 20, 40, 80, 120]
_maxPU = [60, 80, 100, 150, 200, 250]
_minVtx = [0, 80, 120]
_maxVtx = [60, 100, 150, 200, 250]
_maxEff = 1.025
_maxFake = [0.05, 0.1, 0.2, 0.5, 0.7, 1.025]
_minMaxRes = [5, 10, 50, 100, 200, 500, 1000, 2000, 5000]
_minMaxPt = [5e-1, 1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]

_vertexNumberOfEventsHistogram = "DQMData/Run 1/Vertexing/Run summary/PrimaryVertexV/GenPV_Z"

_common = {"xtitle": "Simulated interactions", "xmin": _minPU, "xmax": _maxPU, "ymin": _minVtx, "ymax": _maxVtx}
_recovsgen = PlotGroup("recovsgen", [
    Plot("RecoVtx_vs_GenVtx", ytitle="Reco vertices", **_common),
    Plot("MatchedRecoVtx_vs_GenVtx", ytitle="Matched reco vertices", **_common),
    Plot("merged_vs_ClosestVertexInZ", xtitle="Closest distance in Z (cm)", ytitle="Merge rate", xlog=True, ymax=_maxFake),
    Plot("merged_vs_Z", xtitle="Z (cm)", ytitle="Merge rate", xmin=-20, xmax=20, ymax=_maxFake),
],
                       legendDy=-0.025
)
_pvtagging = PlotGroup("pvtagging", [
    Plot("TruePVLocationIndexCumulative", xtitle="Signal PV status in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, xbinlabels=["Not reconstructed", "Reco and identified", "Reco, not identified"], xbinlabelsize=15, xbinlabeloption="h", xgrid=False, ylog=True, ymin=1e-3, ratioCoverageXrange=[-0.5, 0.5]),
    Plot("TruePVLocationIndex", xtitle="Index of signal PV in reco collection", ytitle="Fraction of events", drawStyle="hist", normalizeToUnitArea=True, ylog=True, ymin=1e-5),
    Plot("MisTagRate_vs_PU", xtitle="PU", ytitle="Mistag rate vs. PU", title="", xmax=_maxPU, ymax=_maxFake),
    Plot("MisTagRate_vs_sum-pt2", xtitle="#Sigmap_{T}^{2}", ytitle="Mistag rate vs. #Sigmap_{T}^{2}", title="", xlog=True, ymax=_maxFake),
],
                       legendDy=-0.025
)
_effandfake = PlotGroup("effandfake", [
    Plot("effic_vs_NumVertices", xtitle="Simulated interactions", ytitle="Efficiency vs. N sim vertices", xmin=_minPU, xmax=_maxPU, ymax=_maxEff),
    Plot("fakerate_vs_PU", xtitle="Simulated interactions", ytitle="Fake rate vs. N sim vertices", xmin=_minPU, xmax=_maxPU, ymax=_maxFake),
    Plot("effic_vs_NumTracks", xtitle="Tracks", ytitle="Efficiency vs. N tracks", title="", ymax=_maxEff),
    Plot("fakerate_vs_NumTracks", xtitle="Tracks", ytitle="Fake rate vs. N tracks", title="", ymax=_maxFake),
    Plot("effic_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Efficiency vs. sum p_{T}^{2}", xlog=True, ymax=_maxEff),
    Plot("fakerate_vs_Pt2", xtitle="Sum p_{T}^{2}    ", ytitle="Fake rate vs. sum p_{T}^{2}", xlog=True, ymax=_maxFake),
])
_common = {"title": "", "stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "ylog": True, "ymin": [5e-7, 5e-6, 5e-5, 5e-4]}
_resolution = PlotGroup("resolution", [
    Plot("RecoAllAssoc2GenMatched_ResolX", xtitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX", xtitle="Resolution in x for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY", xtitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY", xtitle="Resolution in y for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ", xtitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ", xtitle="Resolution in z for merged vertices (#mum)", **_common),
])
_common = dict(title="", xtitle="Number of tracks", scale=1e4, ylog=True, ymin=_minMaxRes , ymax=_minMaxRes)
_resolutionNumTracks = PlotGroup("resolutionNumTracks", [
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_NumTracks_Sigma", ytitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_NumTracks_Sigma", ytitle="Resolution in x for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_NumTracks_Sigma", ytitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_NumTracks_Sigma", ytitle="Resolution in y for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_NumTracks_Sigma", ytitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_NumTracks_Sigma", ytitle="Resolution in z for merged vertices (#mum)", **_common),
])
_common.update(dict(xtitle= "Sum of track p_{T} (GeV)", xlog=True, xmin=_minMaxPt, xmax=_minMaxPt))
_resolutionPt = PlotGroup("resolutionPt", [
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Pt_Sigma", ytitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Pt_Sigma", ytitle="Resolution in x for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Pt_Sigma", ytitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Pt_Sigma", ytitle="Resolution in y for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Pt_Sigma", ytitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Pt_Sigma", ytitle="Resolution in z for merged vertices (#mum)", **_common),
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
                             legendDy=-0.025
)
_common = dict(normalizeToUnitArea=True, drawStyle="HIST", stat=True)
_k0_mass = PlotGroup("mass", [
    Plot("ksMassAll", xtitle="mass of all (GeV)", **_common),
    Plot("ksMassGood", xtitle="mass of good (GeV)", **_common),
    Plot("ksMassFake", xtitle="mass of fake (GeV)", **_common),
],
                     legendDy=-0.025
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
                                 legendDy=-0.025
)
_lambda_mass = PlotGroup("mass", [
    Plot("lamMassAll", xtitle="mass of all (GeV)", **_common),
    Plot("lamMassGood", xtitle="mass of good (GeV)", **_common),
    Plot("lamMassFake", xtitle="mass of fake (GeV)", **_common),
],
                         legendDy=-0.025
)

## Extended set of plots
_common = dict(drawStyle = "HIST", stat=True)
_commonXY = dict(xmin=[x*0.1 for x in xrange(-6, 6, 1)], xmax=[x*0.1 for x in xrange(-5, 7, 1)])
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
_common = dict(title="", xtitle="Vertex z (cm)", scale=1e4, ylog=True, ymin=_minMaxRes , ymax=_minMaxRes, xmin=range(-60,-10,10), xmax=range(20,70,10))
_extResolutionZ = PlotGroup("resolutionZ", [
    Plot("RecoAllAssoc2GenMatched_ResolX_vs_Z_Sigma", ytitle="Resolution in x (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolX_vs_Z_Sigma", ytitle="Resolution in x for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolY_vs_Z_Sigma", ytitle="Resolution in y (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolY_vs_Z_Sigma", ytitle="Resolution in y for merged vertices (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatched_ResolZ_vs_Z_Sigma", ytitle="Resolution in z (#mum)", **_common),
    Plot("RecoAllAssoc2GenMatchedMerged_ResolZ_vs_Z_Sigma", ytitle="Resolution in z for merged vertices (#mum)", **_common),
])
_extDqm = PlotGroup("dqm", [
    Plot("tagVtxTrksVsZ", xtitle="z_{vertex} - z_{beamspot} (cm)", ytitle="Tracks / selected PV"),
    Plot("otherVtxTrksVsZ", xtitle="z_{vertex} - z_{beamspot} (cm)", ytitle="Tracks / pileup vertex"),
    Plot("vtxNbr", xtitle="Reconstructed vertices", ytitle="Events", stat=True, drawStyle="hist", xmin=_minVtx, xmax=_maxVtx),
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
    _puritymissing,
    _sumpt2,
    purpose=PlotPurpose.Vertexing,
    page="vertex",
    onlyForPileup=True
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
    purpose=PlotPurpose.Vertexing,
    page="vertex",
    onlyForPileup=True,
    numberOfEventsHistogram=_vertexNumberOfEventsHistogram
))
plotterExt.append("dqm", _vertexDqmFolders, PlotFolder(
    _extDqm,
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
