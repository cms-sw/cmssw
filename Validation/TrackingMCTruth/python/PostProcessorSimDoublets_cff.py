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

_defaultSubdirsGeneral = ["Tracking/TrackingMCTruth/SimPixelTracks/general"]
_defaultSubdirsTop = ["Tracking/TrackingMCTruth/SimPixelTracks"]
_defaultSubdirsSimNtuplets = ["Tracking/TrackingMCTruth/SimPixelTracks/SimNtuplets/longest", "Tracking/TrackingMCTruth/SimPixelTracks/SimNtuplets/mostAlive"]
_defaultSubdirsSimDoublets = ["Tracking/TrackingMCTruth/SimPixelTracks/SimDoublets"]



##############################################
# General directory efficiencies
##############################################
postProcessorGeneral = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsGeneral),
    efficiency = cms.vstring(
        "eff_vs_pt 'TrackingParticle efficiency (have an alive SimNtuplet); TP transverse momentum p_{T} [GeV]; Efficiency for TrackingParticles' pass_num_vs_pt num_vs_pt",
        "eff_vs_eta 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; Efficiency for TrackingParticles' pass_num_vs_eta num_vs_eta",
        "eff_vs_phi 'TrackingParticle efficiency (have an alive SimNtuplet); TP azimuth angle #phi; Efficiency for TrackingParticles' pass_num_vs_phi num_vs_phi",
    ),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    makeGlobalEffienciesPlot = cms.untracked.bool(True)
)
_addNoFlow(postProcessorGeneral)

postProcessorGeneral2D = DQMEDHarvester("DQMGenericClient",
    makeGlobalEffienciesPlot = cms.untracked.bool(False),
    subDirs = cms.untracked.vstring(_defaultSubdirsGeneral),
    efficiency = cms.vstring(
        "eff_vs_etaPhi 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; TP azimuth angle #phi' pass_num_vs_etaPhi num_vs_etaPhi",
        "eff_vs_etaPt 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; TP transverse momentum p_{T} [GeV]' pass_num_vs_etaPt num_vs_etaPt",
        "eff_vs_phiPt 'TrackingParticle efficiency (have an alive SimNtuplet); TP azimuth angle #phi; TP transverse momentum p_{T} [GeV]' pass_num_vs_phiPt num_vs_phiPt",
        "loss_vs_etaPhi 'TrackingParticle loss rate (have no alive SimNtuplet); TP pseudorapidity #eta; TP azimuth angle #phi' pass_num_vs_etaPhi num_vs_etaPhi fake",
        "loss_vs_etaPt 'TrackingParticle loss rate (have no alive SimNtuplet); TP pseudorapidity #eta; TP transverse momentum p_{T} [GeV]' pass_num_vs_etaPt num_vs_etaPt fake",
        "loss_vs_phiPt 'TrackingParticle loss rate (have no alive SimNtuplet); TP azimuth angle #phi; TP transverse momentum p_{T} [GeV]' pass_num_vs_phiPt num_vs_phiPt fake",
        "eff_vs_nLayersEta 'TrackingParticle efficiency (have an alive SimNtuplet); TP pseudorapidity #eta; Number of layers hit' pass_numLayers_vs_eta numLayers_vs_eta",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)


##############################################
# SimDoublets directory efficiencies
##############################################
postProcessorSimDoublets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsSimDoublets),
    efficiency = cms.vstring(
        "eff_vs_pt 'SimDoublets efficiency vs p_{T}; TP transverse momentum p_{T} [GeV]; Total fraction of SimDoublets passing all cuts' pass_num_vs_pt num_vs_pt",
        "eff_vs_eta 'SimDoublets efficiency vs #eta; TP pseudorapidity #eta; Total fraction of SimDoublets passing all cuts' pass_num_vs_eta num_vs_eta",
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
    subDirs = cms.untracked.vstring(_defaultSubdirsSimDoublets),
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


postProcessorSimNtuplets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsTop),
    efficiency = cms.vstring(
        "simNtuplets/longest/fracAlive_vs_pT 'Fraction of TPs with longest SimNtuplet being alive vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_alive general/numTPVsPt",
        "simNtuplets/longest/fracAlive_vs_eta 'Fraction of TPs with longest SimNtuplet being alive vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_alive general/numTPVsEta",
        "simNtuplets/longest/fracUndefDoubletCuts_vs_pT 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_undefDoubletCuts general/numTPVsPt",
        "simNtuplets/longest/fracUndefDoubletCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_undefDoubletCuts general/numTPVsEta",
        "simNtuplets/longest/fracUndefConnectionCuts_vs_pT 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_undefConnectionCuts general/numTPVsPt",
        "simNtuplets/longest/fracUndefConnectionCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_undefConnectionCuts general/numTPVsEta",
        "simNtuplets/longest/fracMissingLayerPair_vs_pT 'Fraction of TPs with longest SimNtuplet with missing layer pair vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_missingLayerPair general/numTPVsPt",
        "simNtuplets/longest/fracMissingLayerPair_vs_eta 'Fraction of TPs with longest SimNtuplet with missing layer pair vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_missingLayerPair general/numTPVsEta",
        "simNtuplets/longest/fracKilledDoublets_vs_pT 'Fraction of TPs with longest SimNtuplet with killed doublets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_killedDoublets general/numTPVsPt",
        "simNtuplets/longest/fracKilledDoublets_vs_eta 'Fraction of TPs with longest SimNtuplet with killed doublets vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_killedDoublets general/numTPVsEta",
        "simNtuplets/longest/fracKilledConnections_vs_pT 'Fraction of TPs with longest SimNtuplet with killed connections vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_killedConnections general/numTPVsPt",
        "simNtuplets/longest/fracKilledConnections_vs_eta 'Fraction of TPs with longest SimNtuplet with killed connections vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_killedConnections general/numTPVsEta",
        "simNtuplets/longest/fracTooShort_vs_pT 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_tooShort general/numTPVsPt",
        "simNtuplets/longest/fracTooShort_vs_eta 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_tooShort general/numTPVsEta",
        "simNtuplets/longest/fracNotStartingPair_vs_pT 'Fraction of TPs with longest SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/longest/pt_notStartingPair general/numTPVsPt",
        "simNtuplets/longest/fracNotStartingPair_vs_eta 'Fraction of TPs with longest SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/longest/eta_notStartingPair general/numTPVsEta",
        "simNtuplets/mostAlive/fracAlive_vs_pT 'Fraction of TPs with most alive SimNtuplet being alive vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_alive general/numTPVsPt",
        "simNtuplets/mostAlive/fracAlive_vs_eta 'Fraction of TPs with most alive SimNtuplet being alive vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_alive general/numTPVsEta",
        "simNtuplets/mostAlive/fracUndefDoubletCuts_vs_pT 'Fraction of TPs with most alive SimNtuplet having undef doublet cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_undefDoubletCuts general/numTPVsPt",
        "simNtuplets/mostAlive/fracUndefDoubletCuts_vs_eta 'Fraction of TPs with most alive SimNtuplet having undef doublet cuts vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_undefDoubletCuts general/numTPVsEta",
        "simNtuplets/mostAlive/fracUndefConnectionCuts_vs_pT 'Fraction of TPs with most alive SimNtuplet having undef connection cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_undefConnectionCuts general/numTPVsPt",
        "simNtuplets/mostAlive/fracUndefConnectionCuts_vs_eta 'Fraction of TPs with most alive SimNtuplet having undef connection cuts vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_undefConnectionCuts general/numTPVsEta",
        "simNtuplets/mostAlive/fracMissingLayerPair_vs_pT 'Fraction of TPs with most alive SimNtuplet with missing layer pair vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_missingLayerPair general/numTPVsPt",
        "simNtuplets/mostAlive/fracMissingLayerPair_vs_eta 'Fraction of TPs with most alive SimNtuplet with missing layer pair vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_missingLayerPair general/numTPVsEta",
        "simNtuplets/mostAlive/fracKilledDoublets_vs_pT 'Fraction of TPs with most alive SimNtuplet with killed doublets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_killedDoublets general/numTPVsPt",
        "simNtuplets/mostAlive/fracKilledDoublets_vs_eta 'Fraction of TPs with most alive SimNtuplet with killed doublets vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_killedDoublets general/numTPVsEta",
        "simNtuplets/mostAlive/fracKilledConnections_vs_pT 'Fraction of TPs with most alive SimNtuplet with killed connections vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_killedConnections general/numTPVsPt",
        "simNtuplets/mostAlive/fracKilledConnections_vs_eta 'Fraction of TPs with most alive SimNtuplet with killed connections vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_killedConnections general/numTPVsEta",
        "simNtuplets/mostAlive/fracTooShort_vs_pT 'Fraction of TPs with most alive SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_tooShort general/numTPVsPt",
        "simNtuplets/mostAlive/fracTooShort_vs_eta 'Fraction of TPs with most alive SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_tooShort general/numTPVsEta",
        "simNtuplets/mostAlive/fracNotStartingPair_vs_pT 'Fraction of TPs with most alive SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' simNtuplets/mostAlive/pt_tooShort general/numTPVsPt",
        "simNtuplets/mostAlive/fracNotStartingPair_vs_eta 'Fraction of TPs with most alive SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs #eta; TP pseudorapidity #eta; Fraction' simNtuplets/mostAlive/eta_tooShort general/numTPVsEta",
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
        "fracAlive_firstLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being alive; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayerVsEta firstLayerVsEta",
        "fracAlive_lastLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being alive; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayerVsEta lastLayerVsEta",
        "fracAlive_layerSpan 'Fraction of TPs with selected SimNtuplet being alive; First layer ID; Last layer ID; Fraction' pass_layerSpan layerSpan",
        "fracLost_firstLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being lost; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayerVsEta firstLayerVsEta fake",
        "fracLost_lastLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being lost; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayerVsEta lastLayerVsEta fake",
        "fracLost_layerSpan 'Fraction of TPs with selected SimNtuplet being lost; First layer ID; Last layer ID; Fraction' pass_layerSpan layerSpan fake",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)

# _addNoFlow(postProcessorSimNtuplets2D)



postProcessorSimDoubletsSequence = cms.Sequence(
    postProcessorGeneral +
    postProcessorGeneral2D +
    postProcessorSimDoublets +
    postProcessorSimDoublets2D +
    postProcessorSimNtuplets +
    postProcessorSimNtuplets2D
)