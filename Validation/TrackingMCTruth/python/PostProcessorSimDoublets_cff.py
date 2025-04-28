import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

def _addNoFlow(module):
    _noflowSeen = set()
    for eff in module.efficiency.value():
        tmp = eff.split(" ")
        if "cut" in tmp[0]:
            continue
        ind = -1
        if tmp[ind] == "fake" or tmp[ind] == "simpleratio":
            ind = -2
        if not tmp[ind] in _noflowSeen:
            module.noFlowDists.append(tmp[ind])
        if not tmp[ind-1] in _noflowSeen:
            module.noFlowDists.append(tmp[ind-1])

_defaultSubdirsGeneral = ["Tracking/TrackingMCTruth/SimDoublets/general"]
_defaultSubdirsSimNtuplets = ["Tracking/TrackingMCTruth/SimDoublets/simNtuplets"]

postProcessorSimDoublets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsGeneral),
    efficiency = cms.vstring(
        "efficiency_vs_pT 'SimDoublets efficiency vs p_{T}; TP transverse momentum p_{T} [GeV]; Total fraction of SimDoublets passing all cuts' pass_numVsPt numVsPt",
        "efficiency_vs_eta 'SimDoublets efficiency vs #eta; TP pseudorapidity #eta; Total fraction of SimDoublets passing all cuts' pass_numVsEta numVsEta",
        "efficiencyTP_vs_pT 'TrackingParticle efficiency (have an alive SimNtuplet); TP transverse momentum p_{T} [GeV]; Efficiency for TrackingParticles' pass_numTPVsPt numTPVsPt",
        "efficiencyTP_vs_eta 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; Efficiency for TrackingParticles' pass_numTPVsEta numTPVsEta",
        "efficiencyTP_vs_phi 'TrackingParticle efficiency (have an alive SimNtuplet); TP azimuth angle #phi; Efficiency for TrackingParticles' pass_numTPVsPhi numTPVsPhi",
    ),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    makeGlobalEffienciesPlot = cms.untracked.bool(True)
)

_addNoFlow(postProcessorSimDoublets)

postProcessorSimDoublets2D = DQMEDHarvester("DQMGenericClient",
    makeGlobalEffienciesPlot = cms.untracked.bool(False),
    subDirs = cms.untracked.vstring(_defaultSubdirsGeneral),
    efficiency = cms.vstring(
        "efficiency_vs_layerPair 'Total fraction of SimDoublets passing all cuts; Inner layer ID; Outer layer ID' pass_layerPairs layerPairs",
        "efficiencyTP_vs_eta_phi 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; TP azimuth angle #phi' pass_numTPVsEtaPhi numTPVsEtaPhi",
        "efficiencyTP_vs_eta_pT 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; TP transverse momentum p_{T} [GeV]' pass_numTPVsEtaPt numTPVsEtaPt",
        "efficiencyTP_vs_phi_pT 'TrackingParticle efficiency (have an alive SimNtuplet); TP azimuth angle #phi; TP transverse momentum p_{T} [GeV]' pass_numTPVsPhiPt numTPVsPhiPt",
        "lossTP_vs_eta_phi 'TrackingParticle loss rate (have no alive SimNtuplet); TP pseudorapidity #eta; TP azimuth angle #phi' pass_numTPVsEtaPhi numTPVsEtaPhi fake",
        "lossTP_vs_eta_pT 'TrackingParticle loss rate (have no alive SimNtuplet); TP pseudorapidity #eta; TP transverse momentum p_{T} [GeV]' pass_numTPVsEtaPt numTPVsEtaPt fake",
        "lossTP_vs_phi_pT 'TrackingParticle loss rate (have no alive SimNtuplet); TP azimuth angle #phi; TP transverse momentum p_{T} [GeV]' pass_numTPVsPhiPt numTPVsPhiPt fake",
        "efficiencyTP_vs_eta_nLayers 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; Number of layers hit' pass_numLayersVsEta numLayersVsEta",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)

# _addNoFlow(postProcessorSimDoublets2D)

postProcessorSimNtuplets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsSimNtuplets),
    efficiency = cms.vstring(
        "fracAlive_vs_pT 'Fraction of TPs with longest SimNtuplet being alive vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_alive pt",
        "fracAlive_vs_eta 'Fraction of TPs with longest SimNtuplet being alive vs #eta; TP pseudorapidity #eta; Fraction' eta_alive eta",
        "fracUndefDoubletCuts_vs_pT 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_undefDoubletCuts pt",
        "fracUndefDoubletCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs #eta; TP pseudorapidity #eta; Fraction' eta_undefDoubletCuts eta",
        "fracUndefConnectionCuts_vs_pT 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_undefConnectionCuts pt",
        "fracUndefConnectionCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs #eta; TP pseudorapidity #eta; Fraction' eta_undefConnectionCuts eta",
        "fracMissingLayerPair_vs_pT 'Fraction of TPs with longest SimNtuplet with missing layer pair vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_missingLayerPair pt",
        "fracMissingLayerPair_vs_eta 'Fraction of TPs with longest SimNtuplet with missing layer pair vs #eta; TP pseudorapidity #eta; Fraction' eta_missingLayerPair eta",
        "fracKilledDoublets_vs_pT 'Fraction of TPs with longest SimNtuplet with killed doublets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_killedDoublets pt",
        "fracKilledDoublets_vs_eta 'Fraction of TPs with longest SimNtuplet with killed doublets vs #eta; TP pseudorapidity #eta; Fraction' eta_killedDoublets eta",
        "fracKilledConnections_vs_pT 'Fraction of TPs with longest SimNtuplet with killed connections vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_killedConnections pt",
        "fracKilledConnections_vs_eta 'Fraction of TPs with longest SimNtuplet with killed connections vs #eta; TP pseudorapidity #eta; Fraction' eta_killedConnections eta",
        "fracTooShort_vs_pT 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' pt_tooShort pt",
        "fracTooShort_vs_eta 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs #eta; TP pseudorapidity #eta; Fraction' eta_tooShort eta",
    ),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    makeGlobalEffienciesPlot = cms.untracked.bool(True)
)

_addNoFlow(postProcessorSimNtuplets)

postProcessorSimNtuplets2D = DQMEDHarvester("DQMGenericClient",
    makeGlobalEffienciesPlot = cms.untracked.bool(False),
    subDirs = cms.untracked.vstring(_defaultSubdirsSimNtuplets),
    efficiency = cms.vstring(
        "fracAlive_firstLayer_vs_eta 'Fraction of TPs with longest SimNtuplet being alive; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayerVsEta firstLayerVsEta",
        "fracAlive_lastLayer_vs_eta 'Fraction of TPs with longest SimNtuplet being alive; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayerVsEta lastLayerVsEta",
        "fracAlive_layerSpan 'Fraction of TPs with longest SimNtuplet being alive; First layer ID; Last layer ID; Fraction' pass_layerSpan layerSpan",
        "fracLost_firstLayer_vs_eta 'Fraction of TPs with longest SimNtuplet being lost; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayerVsEta firstLayerVsEta fake",
        "fracLost_lastLayer_vs_eta 'Fraction of TPs with longest SimNtuplet being lost; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayerVsEta lastLayerVsEta fake",
        "fracLost_layerSpan 'Fraction of TPs with longest SimNtuplet being lost; First layer ID; Last layer ID; Fraction' pass_layerSpan layerSpan fake",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)

# _addNoFlow(postProcessorSimNtuplets2D)



postProcessorSimDoubletsSequence = cms.Sequence(
    postProcessorSimDoublets +
    postProcessorSimDoublets2D +
    postProcessorSimNtuplets +
    postProcessorSimNtuplets2D
)