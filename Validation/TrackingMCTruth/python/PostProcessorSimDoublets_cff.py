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
        "eff_vs_dxy 'TrackingParticle efficiency (have an alive SimNtuplet); TP transverse IP to beamspot dxy [cm]; Efficiency for TrackingParticles' pass_num_vs_dxy num_vs_dxy",
        "eff_vs_dz 'TrackingParticle efficiency (have an alive SimNtuplet); TP longitudinal IP to beamspot dz [cm]; Efficiency for TrackingParticles' pass_num_vs_dz num_vs_dz",
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
        "eff_vs_layerPair 'Total fraction of SimDoublets passing all cuts; Inner layer ID; Outer layer ID' pass_layerPairs layerPairs",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)


postProcessorSimNtuplets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsTop),
    efficiency = cms.vstring(
        "SimNtuplets/longest/fracAlive_vs_pt 'Fraction of TPs with longest SimNtuplet being alive vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_alive general/num_vs_pt",
        "SimNtuplets/longest/fracAlive_vs_eta 'Fraction of TPs with longest SimNtuplet being alive vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_alive general/num_vs_eta",
        "SimNtuplets/longest/fracUndefDoubletCuts_vs_pt 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_undefDoubletCuts general/num_vs_pt",
        "SimNtuplets/longest/fracUndefDoubletCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef doublet cuts vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_undefDoubletCuts general/num_vs_eta",
        "SimNtuplets/longest/fracUndefConnectionCuts_vs_pt 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_undefConnectionCuts general/num_vs_pt",
        "SimNtuplets/longest/fracUndefConnectionCuts_vs_eta 'Fraction of TPs with longest SimNtuplet having undef connection cuts vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_undefConnectionCuts general/num_vs_eta",
        "SimNtuplets/longest/fracMissingLayerPair_vs_pt 'Fraction of TPs with longest SimNtuplet with missing layer pair vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_missingLayerPair general/num_vs_pt",
        "SimNtuplets/longest/fracMissingLayerPair_vs_eta 'Fraction of TPs with longest SimNtuplet with missing layer pair vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_missingLayerPair general/num_vs_eta",
        "SimNtuplets/longest/fracKilledDoublets_vs_pt 'Fraction of TPs with longest SimNtuplet with killed doublets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_killedDoublets general/num_vs_pt",
        "SimNtuplets/longest/fracKilledDoublets_vs_eta 'Fraction of TPs with longest SimNtuplet with killed doublets vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_killedDoublets general/num_vs_eta",
        "SimNtuplets/longest/fracKilledConnections_vs_pt 'Fraction of TPs with longest SimNtuplet with killed connections vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_killedConnections general/num_vs_pt",
        "SimNtuplets/longest/fracKilledConnections_vs_eta 'Fraction of TPs with longest SimNtuplet with killed connections vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_killedConnections general/num_vs_eta",
        "SimNtuplets/longest/fracTooShort_vs_pt 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_tooShort general/num_vs_pt",
        "SimNtuplets/longest/fracTooShort_vs_eta 'Fraction of TPs with longest SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_tooShort general/num_vs_eta",
        "SimNtuplets/longest/fracNotStartingPair_vs_pt 'Fraction of TPs with longest SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/longest/num_pt_notStartingPair general/num_vs_pt",
        "SimNtuplets/longest/fracNotStartingPair_vs_eta 'Fraction of TPs with longest SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/longest/num_eta_notStartingPair general/num_vs_eta",
        "SimNtuplets/mostAlive/fracAlive_vs_pt 'Fraction of TPs with most alive SimNtuplet being alive vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_alive general/num_vs_pt",
        "SimNtuplets/mostAlive/fracAlive_vs_eta 'Fraction of TPs with most alive SimNtuplet being alive vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_alive general/num_vs_eta",
        "SimNtuplets/mostAlive/fracUndefDoubletCuts_vs_pt 'Fraction of TPs with most alive SimNtuplet having undef doublet cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_undefDoubletCuts general/num_vs_pt",
        "SimNtuplets/mostAlive/fracUndefDoubletCuts_vs_eta 'Fraction of TPs with most alive SimNtuplet having undef doublet cuts vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_undefDoubletCuts general/num_vs_eta",
        "SimNtuplets/mostAlive/fracUndefConnectionCuts_vs_pt 'Fraction of TPs with most alive SimNtuplet having undef connection cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_undefConnectionCuts general/num_vs_pt",
        "SimNtuplets/mostAlive/fracUndefConnectionCuts_vs_eta 'Fraction of TPs with most alive SimNtuplet having undef connection cuts vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_undefConnectionCuts general/num_vs_eta",
        "SimNtuplets/mostAlive/fracMissingLayerPair_vs_pt 'Fraction of TPs with most alive SimNtuplet with missing layer pair vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_missingLayerPair general/num_vs_pt",
        "SimNtuplets/mostAlive/fracMissingLayerPair_vs_eta 'Fraction of TPs with most alive SimNtuplet with missing layer pair vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_missingLayerPair general/num_vs_eta",
        "SimNtuplets/mostAlive/fracKilledDoublets_vs_pt 'Fraction of TPs with most alive SimNtuplet with killed doublets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_killedDoublets general/num_vs_pt",
        "SimNtuplets/mostAlive/fracKilledDoublets_vs_eta 'Fraction of TPs with most alive SimNtuplet with killed doublets vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_killedDoublets general/num_vs_eta",
        "SimNtuplets/mostAlive/fracKilledConnections_vs_pt 'Fraction of TPs with most alive SimNtuplet with killed connections vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_killedConnections general/num_vs_pt",
        "SimNtuplets/mostAlive/fracKilledConnections_vs_eta 'Fraction of TPs with most alive SimNtuplet with killed connections vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_killedConnections general/num_vs_eta",
        "SimNtuplets/mostAlive/fracTooShort_vs_pt 'Fraction of TPs with most alive SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_tooShort general/num_vs_pt",
        "SimNtuplets/mostAlive/fracTooShort_vs_eta 'Fraction of TPs with most alive SimNtuplet having 3 RecHits but yet being to short to pass the threshold vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_tooShort general/num_vs_eta",
        "SimNtuplets/mostAlive/fracNotStartingPair_vs_pt 'Fraction of TPs with most alive SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/mostAlive/num_pt_notStartingPair general/num_vs_pt",
        "SimNtuplets/mostAlive/fracNotStartingPair_vs_eta 'Fraction of TPs with most alive SimNtuplet starting in a layer pair not considered as a starting point for Ntuplets vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/mostAlive/num_eta_notStartingPair general/num_vs_eta",
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
        "fracAlive_firstLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being alive; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayer_vs_eta firstLayer_vs_eta",
        "fracAlive_lastLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being alive; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayer_vs_eta lastLayer_vs_eta",
        "fracAlive_layerSpan 'Fraction of TPs with selected SimNtuplet being alive; First layer ID; Last layer ID; Fraction' pass_layerSpan layerSpan",
        "fracLost_firstLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being lost; TP pseudorapidity #eta; First layer ID; Fraction' pass_firstLayer_vs_eta firstLayer_vs_eta fake",
        "fracLost_lastLayer_vs_eta 'Fraction of TPs with selected SimNtuplet being lost; TP pseudorapidity #eta; Last layer ID; Fraction' pass_lastLayer_vs_eta lastLayer_vs_eta fake",
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