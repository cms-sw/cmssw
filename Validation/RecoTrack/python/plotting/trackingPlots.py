from plotting import FakeDuplicate, AggregateBins, Plot, PlotGroup, Plotter, AlgoOpt
import validation

_maxEff = [0.2, 0.5, 0.8, 1.025]
_maxFake = [0.2, 0.5, 0.8, 1.025]

_effandfake1 = PlotGroup("effandfake1", [
    Plot("effic", xtitle="#eta", ytitle="efficiency vs #eta", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_eta", assoc="num_assoc(recoToSim)_eta", dup="num_duplicate_eta", reco="num_reco_eta", title="fake+duplicates vs #eta"),
         xtitle="#eta", ytitle="fake+duplicates vs #eta", ymax=_maxFake),
    Plot("efficPt", title="", xtitle="p_{t}", ytitle="efficiency vs p_{t}", xmax=300, xlog=True),
    Plot(FakeDuplicate("fakeduprate_vs_pT", assoc="num_assoc(recoToSim)_pT", dup="num_duplicate_pT", reco="num_reco_pT", title=""),
         xtitle="p_{t}", ytitle="fake+duplicates rate vs p_{t}", ymax=_maxFake, xmin=0.2, xmax=300, xlog=True),
    Plot("effic_vs_hit", xtitle="hits", ytitle="efficiency vs hits"),
    Plot(FakeDuplicate("fakeduprate_vs_hit", assoc="num_assoc(recoToSim)_hit", dup="num_duplicate_hit", reco="num_reco_hit", title="fake+duplicates vs hit"),
         xtitle="hits", ytitle="fake+duplicates rate vs hits", ymax=_maxFake),
])
_effandfake2 = PlotGroup("effandfake2", [
    Plot("effic_vs_phi", xtitle="#phi", ytitle="efficiency vs #phi", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_phi", assoc="num_assoc(recoToSim)_phi", dup="num_duplicate_phi", reco="num_reco_phi", title="fake+duplicates vs #phi"),
         xtitle="#phi", ytitle="fake+duplicates rate vs #phi", ymax=_maxFake),
    Plot("effic_vs_dxy", title="", xtitle="dxy", ytitle="efficiency vs dxy", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dxy", assoc="num_assoc(recoToSim)_dxy", dup="num_duplicate_dxy", reco="num_reco_dxy", title=""),
         xtitle="dxy", ytitle="fake+duplicates rate vs dxy", ymax=_maxFake),
    Plot("effic_vs_dz", xtitle="dz", ytitle="efficiency vs dz", ymax=_maxEff),
    Plot(FakeDuplicate("fakeduprate_vs_dz", assoc="num_assoc(recoToSim)_dz", dup="num_duplicate_dz", reco="num_reco_dz", title="fake+duplicates vs dz"),
         xtitle="dz", ytitle="fake+duplicates rate vs dz", ymax=_maxFake),
])

_dupandfake1 = PlotGroup("dupandfake1", [
    Plot("fakerate", xtitle="#eta", ytitle="fakerate vs #eta", ymax=_maxFake),
    Plot("duplicatesRate", xtitle="#eta", ytitle="duplicates rate vs #eta", ymax=_maxFake),
    Plot("fakeratePt", xtitle="p_{t}", ytitle="fakerate vs p_{t}", xmax=300, xlog=True, ymax=_maxFake),
    Plot("duplicatesRate_Pt", title="", xtitle="p_{t}", ytitle="duplicates rate vs p_{t}", xmin=0.2, xmax=300, ymax=_maxFake, xlog=True),
    Plot("fakerate_vs_hit", xtitle="hits", ytitle="fakerate vs hits", ymax=_maxFake),
    Plot("duplicatesRate_hit", xtitle="hits", ytitle="duplicates rate vs hits", ymax=_maxFake)
])
_dupandfake2 = PlotGroup("dupandfake2", [
    Plot("fakerate_vs_phi", xtitle="#phi", ytitle="fakerate vs #phi", ymax=_maxFake),
    Plot("duplicatesRate_phi", xtitle="#phi", ytitle="duplicates rate vs #phi", ymax=_maxFake),
    Plot("fakerate_vs_dxy", xtitle="dxy", ytitle="fakerate vs dxy", ymax=_maxFake),
    Plot("duplicatesRate_dxy", title="", xtitle="dxy", ytitle="duplicates rate vs dxy", ymax=_maxFake),
    Plot("fakerate_vs_dz", xtitle="dz", ytitle="fakerate vs dz", ymax=_maxFake),
    Plot("duplicatesRate_dz", xtitle="dz", ytitle="duplicates rate vs dz", ymax=_maxFake),
])

_common = {"ymin": 0, "ymax": 1.025}
_effvspos = PlotGroup("effvspos", [
    Plot("effic_vs_vertpos", xtitle="TP vert xy pos", ytitle="efficiency vs vert xy pos", **_common),
    Plot("effic_vs_zpos", xtitle="TP vert z pos", ytitle="efficiency vs vert z pos", **_common),
    Plot("effic_vs_dr", xlog=True, xtitle="#DeltaR", ytitle="efficiency vs #DeltaR", **_common),
    Plot("fakerate_vs_dr", xlog=True, title="", xtitle="#DeltaR", ytitle="Fake rate vs #DeltaR", ymin=0, ymax=_maxFake)
],
                      legendDy=-0.025
)

_common = {"stat": True, "drawStyle": "hist"}
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
    Plot("chargeMisIdRate_Pt", xtitle="p_{t}", ytitle="charge mis-id rate vs p_{t}", xmax=300, ymax=0.1, xlog=True),
    Plot("chargeMisIdRate_hit", xtitle="hits", ytitle="charge mis-id rate vs hits", title=""),
    Plot("chargeMisIdRate_phi", xtitle="#phi", ytitle="charge mis-id rate vs #phi", title="", ymax=0.01),
    Plot("chargeMisIdRate_dxy", xtitle="dxy", ytitle="charge mis-id rate vs dxy", ymax=0.1),
    Plot("chargeMisIdRate_dz", xtitle="dz", ytitle="charge mis-id rate vs dz", ymax=0.1)
])
_hitsAndPt = PlotGroup("hitsAndPt", [
    Plot("missing_inner_layers", stat=True, normalizeToUnitArea=True, drawStyle="hist"),
    Plot("missing_outer_layers", stat=True, normalizeToUnitArea=True, drawStyle="hist"),
    Plot("nhits_vs_eta", stat=True, statx=0.38, profileX=True, xtitle="#eta", ytitle="<hits> vs #eta", ymin=8, ymax=24, statyadjust=[0,0,-0.15]),
    Plot("hits", stat=True, xtitle="hits", xmin=0, xmax=40, drawStyle="hist"),
    Plot("num_simul_pT", stat=True, normalizeToUnitArea=True, xtitle="p_{t}", xmin=0, xmax=10, drawStyle="hist"),
    Plot("num_reco_pT", stat=True, normalizeToUnitArea=True, xtitle="p_{t}", xmin=0, xmax=10, drawStyle="hist")
])
_common = {"stat": True, "normalizeToUnitArea": True, "drawStyle": "hist"}
_ntracks = PlotGroup("ntracks", [
#    Plot("num_simul_eta", xtitle="#eta", **_common),
#    Plot("num_reco_eta", xtitle="#eta", **_common),
    Plot("num_simul_dr", xtitle="#DeltaR", **_common),
    Plot("num_reco_dr", xtitle="#DeltaR", **_common),
    Plot("num_simul_dxy", xtitle="dxy", **_common),
    Plot("num_reco_dxy", xtitle="dxy", **_common),
    Plot("num_simul_dz", xtitle="dz", **_common),
    Plot("num_reco_dz", xtitle="dz", **_common),
],
#                     legendDy=-0.025
                            legendDy=-0.02, legendDh=-0.01
)
_tuning = PlotGroup("tuning", [
    Plot("chi2", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="#chi^{2}"),
    Plot("chi2_prob", stat=True, normalizeToUnitArea=True, drawStyle="hist", xtitle="Prob(#chi^{2})"),
    Plot("chi2_vs_eta", stat=True, profileX=True, title="", xtitle="#eta", ytitle="< #chi^{2} / ndf >", ymax=2.5),
    Plot("ptres_vs_eta_Mean", stat=True, scale=100, title="", xtitle="#eta", ytitle="< #delta p_{t} / p_{t} > [%]", ymin=-1.5, ymax=1.5)
])
_common = {"stat": True, "fit": True, "normalizeToUnitArea": True, "drawStyle": "hist", "drawCommand": "", "xmin": -10, "xmax": 10, "ylog": True, "ymin": 5e-5, "ymax": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.025]}
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
    Plot("ptres_vs_eta_Sigma", ytitle="#sigma(#delta p_{t}/p_{t})", ymin=0.0059, ymax=0.08, **_common),
],
                            legendDy=-0.02, legendDh=-0.01
)
_common = {"title": "", "ylog": True, "xlog": True, "xtitle": "p_{t}", "xmin": 0.1, "xmax": 1000}
_resolutionsPt = PlotGroup("resolutionsPt", [
    Plot("phires_vs_pt_Sigma", ytitle="#sigma(#delta #phi) [rad]", ymin=0.000009, ymax=0.01, **_common),
    Plot("cotThetares_vs_pt_Sigma", ytitle="#sigma(#delta cot(#theta))", ymin=0.00009, ymax=0.03, **_common),
    Plot("dxyres_vs_pt_Sigma", ytitle="#sigma(#delta d_{0}) [cm]", ymin=0.00009, ymax=0.05, **_common),
    Plot("dzres_vs_pt_Sigma", ytitle="#sigma(#delta z_{0}) [cm]", ymin=0.0009, ymax=0.1, **_common),
    Plot("ptres_vs_pt_Sigma", ytitle="#sigma(#delta p_{t}/p_{t})", ymin=0.003, ymax=2.2, **_common),
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
    _dupandfake1,
    _dupandfake2,
    _effvspos,
    _dedx,
    _chargemisid,
    _hitsAndPt,
    _ntracks,
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
