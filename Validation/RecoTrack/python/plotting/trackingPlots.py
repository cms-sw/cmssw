from plotting import FakeDuplicate, AggregateBins, Plot, PlotGroup, Plotter, AlgoOpt
import validation

_maxEff = [0.2, 0.5, 0.8, 1.025]
_maxFake = [0.2, 0.5, 0.8, 1.025]

_effandfake1 = PlotGroup("effandfake1", [
    Plot("efficPt", title="Efficiency vs p_{T}", xtitle="TP p_{T} (GeV)", ytitle="efficiency vs p_{T}", xlog=True),
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
    Plot("effic_vs_dz", xtitle="TP dz (cm)", ytitle="Efficiency vs dz", title="", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dz", assoc="num_assoc(recoToSim)_dz", dup="num_duplicate_dz", reco="num_reco_dz", title=""),
         xtitle="track dz (cm)", ytitle="fake+duplicates rate vs dz", ymax=_maxFake),
    Plot("effic_vs_hit", xtitle="TP hits", ytitle="efficiency vs hits"),
    Plot(FakeDuplicate("fakeduprate_vs_hit", assoc="num_assoc(recoToSim)_hit", dup="num_duplicate_hit", reco="num_reco_hit", title="fake+duplicates vs hit"),
         xtitle="track hits", ytitle="fake+duplicates rate vs hits", ymax=_maxFake),
])
_effandfake3 = PlotGroup("effandfake3", [
    Plot("effic_vs_layer", xtitle="TP layers", ytitle="efficiency vs layers", xmax=25),
    Plot(FakeDuplicate("fakeduprate_vs_layer", assoc="num_assoc(recoToSim)_layer", dup="num_duplicate_layer", reco="num_reco_layer", title="fake+duplicates vs layer"),
         xtitle="track layers", ytitle="fake+duplicates rate vs layers", ymax=_maxFake, xmax=25),
    Plot("effic_vs_pixellayer", xtitle="TP pixel layers", ytitle="efficiency vs pixel layers", title="", xmax=6),
    Plot(FakeDuplicate("fakeduprate_vs_pixellayer", assoc="num_assoc(recoToSim)_pixellayer", dup="num_duplicate_pixellayer", reco="num_reco_pixellayer", title=""),
         xtitle="track pixel layers", ytitle="fake+duplicates rate vs pixel layers", ymax=_maxFake, xmax=6),
    Plot("effic_vs_3Dlayer", xtitle="TP 3D layers", ytitle="efficiency vs 3D layers", xmax=20),
    Plot(FakeDuplicate("fakeduprate_vs_3Dlayer", assoc="num_assoc(recoToSim)_3Dlayer", dup="num_duplicate_3Dlayer", reco="num_reco_3Dlayer", title="fake+duplicates vs 3D layer"),
         xtitle="track 3D layers", ytitle="fake+duplicates rate vs 3D layers", ymax=_maxFake, xmax=20),
])
_common = {"ymin": 0, "ymax": 1.025}
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
    Plot("fakerate_vs_dz", xtitle="track dz (cm)", ytitle="fakerate vs dz", title="", ymax=_maxFake),
    Plot("duplicatesRate_dz", xtitle="track dz (cm)", ytitle="duplicates rate vs dz", title="", ymax=_maxFake),
    Plot("pileuprate_dz", xtitle="track dz (cm)", ytitle="pileup rate vs dz", title="", ymax=_maxFake),
    Plot("fakerate_vs_hit", xtitle="track hits", ytitle="fakerate vs hits", ymax=_maxFake),
    Plot("duplicatesRate_hit", xtitle="track hits", ytitle="duplicates rate vs hits", ymax=_maxFake),
    Plot("pileuprate_hit", xtitle="track hits", ytitle="pileup rate vs hits", ymax=_maxFake)
], ncols=3)
_dupandfake3 = PlotGroup("dupandfake3", [
    Plot("fakerate_vs_layer", xtitle="track layers", ytitle="fakerate vs layer", ymax=_maxFake, xmax=25),
    Plot("duplicatesRate_layer", xtitle="track layers", ytitle="duplicates rate vs layers", ymax=_maxFake, xmax=25),
    Plot("pileuprate_layer", xtitle="track layers", ytitle="pileup rate vs layers", ymax=_maxFake, xmax=25),
    Plot("fakerate_vs_pixellayer", xtitle="track pixel layers", ytitle="fakerate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    Plot("duplicatesRate_pixellayer", xtitle="track pixel layers", ytitle="duplicates rate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    Plot("pileuprate_pixellayer", xtitle="track pixel layers", ytitle="pileup rate vs pixel layers", title="", ymax=_maxFake, xmax=6),
    Plot("fakerate_vs_3Dlayer", xtitle="track 3D layers", ytitle="fakerate vs 3D layers", ymax=_maxFake, xmax=20),
    Plot("duplicatesRate_3Dlayer", xtitle="track 3D layers", ytitle="duplicates rate vs 3D layers", ymax=_maxFake, xmax=20),
    Plot("pileuprate_3Dlayer", xtitle="track 3D layers", ytitle="pileup rate vs 3D layers", ymax=_maxFake, xmax=20)
], ncols=3)
_dupandfake4 = PlotGroup("dupandfake4", [
    Plot("fakerate_vs_chi2", xtitle="track #chi^{2}", ytitle="fakerate vs #chi^{2}", ymax=_maxFake),
    Plot("duplicatesRate_chi2", xtitle="track #chi^{2}", ytitle="duplicates rate vs #chi^{2}", ymax=_maxFake),
    Plot("pileuprate_chi2", xtitle="track #chi^{2}", ytitle="pileup rate vs #chi^{2}", ymax=_maxFake)
],
                         legendDy=-0.025
)

# These don't exist in FastSim
_common = {"stat": True, "drawStyle": "hist", "ignoreIfMissing": True}
_dedx = PlotGroup("dedx", [
    Plot("h_dedx_estim1", normalizeToUnitArea=True, xtitle="dE/dx, harm2", **_common),
    Plot("h_dedx_estim2", normalizeToUnitArea=True, xtitle="dE/dx, trunc40", **_common),
    Plot("h_dedx_nom1", **_common),
    Plot("h_dedx_sat1", **_common),
    ],
                  legendDy=-0.35
)

_chargemisid = PlotGroup("chargemisid", [
    Plot("chargeMisIdRate", xtitle="#eta", ytitle="charge mis-id rate vs #eta", ymax=0.05),
    Plot("chargeMisIdRate_Pt", xtitle="p_{T}", ytitle="charge mis-id rate vs p_{T}", xmax=300, ymax=0.1, xlog=True),
    Plot("chargeMisIdRate_hit", xtitle="hits", ytitle="charge mis-id rate vs hits", title=""),
    Plot("chargeMisIdRate_phi", xtitle="#phi", ytitle="charge mis-id rate vs #phi", title="", ymax=0.01),
    Plot("chargeMisIdRate_dxy", xtitle="dxy", ytitle="charge mis-id rate vs dxy", ymax=0.1),
    Plot("chargeMisIdRate_dz", xtitle="dz", ytitle="charge mis-id rate vs dz", ymax=0.1)
])
_hitsAndPt = PlotGroup("hitsAndPt", [
    Plot("missing_inner_layers", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=1, drawStyle="hist"),
    Plot("missing_outer_layers", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=1, drawStyle="hist"),
    Plot("hits_eta", stat=True, statx=0.38, xtitle="track #eta", ytitle="<hits> vs #eta", ymin=8, ymax=24, statyadjust=[0,0,-0.15]),
    Plot("hits", stat=True, xtitle="track hits", xmin=0, xmax=40, drawStyle="hist"),
    Plot("num_simul_pT", stat=True, normalizeToUnitArea=True, xtitle="TP p_{T}", xlog=True, drawStyle="hist"),
    Plot("num_reco_pT", stat=True, normalizeToUnitArea=True, xtitle="track p_{T}", xlog=True, drawStyle="hist")
])
_tuning = PlotGroup("tuning", [
    Plot("chi2", stat=True, normalizeToUnitArea=True, ylog=True, ymin=1e-6, ymax=[0.1, 0.2, 0.5, 1.0001], drawStyle="hist", xtitle="#chi^{2}", ratioUncertainty=False),
    Plot("chi2_prob", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="Prob(#chi^{2})", ratioUncertainty=False),
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
_common = {"title": "", "ylog": True, "xtitle": "#eta"}
_resolutionsEta = PlotGroup("resolutionsEta", [
    Plot("phires_vs_eta_Sigma", ytitle="#sigma(#delta #phi) [rad]", ymin=0.000009, ymax=0.01, **_common),
    Plot("cotThetares_vs_eta_Sigma", ytitle="#sigma(#delta cot(#theta))", ymin=0.00009, ymax=0.03, **_common),
    Plot("dxyres_vs_eta_Sigma", ytitle="#sigma(#delta d_{0}) [cm]", ymin=0.00009, ymax=0.05, **_common),
    Plot("dzres_vs_eta_Sigma", ytitle="#sigma(#delta z_{0}) [cm]", ymin=0.0009, ymax=0.1, **_common),
    Plot("ptres_vs_eta_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", ymin=0.0059, ymax=0.08, **_common),
],
                            legendDy=-0.02, legendDh=-0.01
)
_common = {"title": "", "ylog": True, "xlog": True, "xtitle": "p_{T}", "xmin": 0.1, "xmax": 1000}
_resolutionsPt = PlotGroup("resolutionsPt", [
    Plot("phires_vs_pt_Sigma", ytitle="#sigma(#delta #phi) [rad]", ymin=0.000009, ymax=0.01, **_common),
    Plot("cotThetares_vs_pt_Sigma", ytitle="#sigma(#delta cot(#theta))", ymin=0.00009, ymax=0.03, **_common),
    Plot("dxyres_vs_pt_Sigma", ytitle="#sigma(#delta d_{0}) [cm]", ymin=0.00009, ymax=0.05, **_common),
    Plot("dzres_vs_pt_Sigma", ytitle="#sigma(#delta z_{0}) [cm]", ymin=0.0009, ymax=0.1, **_common),
    Plot("ptres_vs_pt_Sigma", ytitle="#sigma(#delta p_{T}/p_{T})", ymin=0.003, ymax=2.2, **_common),
],
                            legendDy=-0.02, legendDh=-0.01
)

plotter = Plotter([
    "DQMData/Run 1/Tracking/Run summary/Track",
    "DQMData/Tracking/Track",
    "DQMData/Run 1/RecoTrackV/Run summary/Track",
    "DQMData/RecoTrackV/Track",
],[
    _effandfake1,
    _effandfake2,
    _effandfake3,
    _effvspos,
    _dupandfake1,
    _dupandfake2,
    _dupandfake3,
    _dupandfake4,
    _dedx,
#    _chargemisid,
    _hitsAndPt,
    _tuning,
    _pulls,
    _resolutionsEta,
    _resolutionsPt,
])

import collections
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


_timing = PlotGroup("timing", [
    Plot(AggregateBins("iterative", "reconstruction_step_module_average", _iterModuleMap), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST", xbinlabelsize=0.03),
#    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap), ytitle="Average processing time", title="Average processing time / event (normalized)", drawStyle="HIST", xbinlabelsize=0.03, normalizeToUnitArea=True)
    Plot(AggregateBins("iterative_norm", "reconstruction_step_module_average", _iterModuleMap, normalizeTo="ak7CaloJets"), ytitle="Average processing time / ak7CaloJets", title="Average processing time / event (normalized to ak7CaloJets)", drawStyle="HIST", xbinlabelsize=0.03)

    ],
                    legendDx=-0.1, legendDw=-0.35, legendDy=0.39,
)
_pixelTiming = PlotGroup("pixelTiming", [
    Plot(AggregateBins("pixel", "reconstruction_step_module_average", {"pixelTracks": ["pixelTracks"]}), ytitle="Average processing time [ms]", title="Average processing time / event", drawStyle="HIST")
])

timePlotter = Plotter([
    "DQMData/Run 1/DQM/Run summary/TimerService/Paths",
    "DQMData/Run 1/DQM/Run summary/TimerService/process RECO/Paths",
],[
    _timing
#    _pixelTiming
]
)

_common = {"stat": True, "normalizeToUnitArea": True, "drawStyle": "hist"}
_tplifetime = PlotGroup("tplifetime", [
    Plot("TPlip", xtitle="TP lip", **_common),
    Plot("TPtip", xtitle="TP tip", **_common),
])

tpPlotter = Plotter([
    "DQMData/Run 1/Tracking/Run summary/TrackingMCTruth/TrackingParticle",
    "DQMData/Tracking/TrackingMCTruth/TrackingParticle",
], [
    _tplifetime,
])


_tracks_map = {
    '': { # all tracks
        'ootb'                : 'general_trackingParticleRecoAsssociation',
        'initialStep'         : 'cutsRecoInitialStep_trackingParticleRecoAsssociation',
        'lowPtTripletStep'    : 'cutsRecoLowPtTripletStep_trackingParticleRecoAsssociation',
        'pixelPairStep'       : 'cutsRecoPixelPairStep_trackingParticleRecoAsssociation',
        'detachedTripletStep' : 'cutsRecoDetachedTripletStep_trackingParticleRecoAsssociation',
        'mixedTripletStep'    : 'cutsRecoMixedTripletStep_trackingParticleRecoAsssociation',
        'pixelLessStep'       : 'cutsRecoPixelLessStep_trackingParticleRecoAsssociation',
        'tobTecStep'          : 'cutsRecoTobTecStep_trackingParticleRecoAsssociation',
        'jetCoreRegionalStep' : 'cutsRecoJetCoreRegionalStep_trackingParticleRecoAsssociation',
        'muonSeededStepInOut' : 'cutsRecoMuonSeededStepInOut_trackingParticleRecoAsssociation',
        'muonSeededStepOutIn' : 'cutsRecoMuonSeededStepOutIn_trackingParticleRecoAsssociation'
    },
    "highPurity": {
        'ootb'                : 'cutsRecoHp_trackingParticleRecoAsssociation',
        'initialStep'         : 'cutsRecoInitialStepHp_trackingParticleRecoAsssociation',
        'lowPtTripletStep'    : 'cutsRecoLowPtTripletStepHp_trackingParticleRecoAsssociation',
        'pixelPairStep'       : 'cutsRecoPixelPairStepHp_trackingParticleRecoAsssociation',
        'detachedTripletStep' : 'cutsRecoDetachedTripletStepHp_trackingParticleRecoAsssociation',
        'mixedTripletStep'    : 'cutsRecoMixedTripletStepHp_trackingParticleRecoAsssociation',
        'pixelLessStep'       : 'cutsRecoPixelLessStepHp_trackingParticleRecoAsssociation',
        'tobTecStep'          : 'cutsRecoTobTecStepHp_trackingParticleRecoAsssociation',
        'jetCoreRegionalStep' : 'cutsRecoJetCoreRegionalStepHp_trackingParticleRecoAsssociation',
        'muonSeededStepInOut' : 'cutsRecoMuonSeededStepInOutHp_trackingParticleRecoAsssociation',
        'muonSeededStepOutIn' : 'cutsRecoMuonSeededStepOutInHp_trackingParticleRecoAsssociation'
    }
}


class TrackingValidation(validation.Validation):
    def _init__(self, *args, **kwargs):
        super(TrackingValidation, self).__init__(*args, **kwargs)

    def _getDirectoryName(self, quality, algo):
        return _tracks_map[quality][algo]

    def _getSelectionName(self, quality, algo):
        ret = ""
        if quality != "":
            ret += "_"+quality
        if not (algo == "ootb" and quality != ""):
            ret += "_"+algo

        return ret
