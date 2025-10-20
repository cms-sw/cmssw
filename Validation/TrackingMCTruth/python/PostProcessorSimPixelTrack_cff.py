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

def buildSimNtupletFractions(subfolder, simNtupletName):
    output = []
    for statusTag, statusLabel in [("Alive", "being alive"), 
                                   ("UndefDoubletCuts", "having undef doublet cuts"),
                                   ("UndefConnectionCuts", "having undef connection cuts"),
                                   ("MissingLayerPair", "with missing layer pair"),
                                   ("KilledDoublets", "with killed doublets"),
                                   ("KilledConnections", "with killed connections"),
                                   ("KilledTripletConnections", "with killed triplet connections"),
                                   ("TooShort", "having 3 RecHits but yet being to short to pass the threshold"),
                                   ("NotStartingPair", "starting in a layer pair not considered as a starting point for Ntuplets")]:
        strings = (subfolder, statusTag, simNtupletName, statusLabel, subfolder, statusTag)
        output.append("SimNtuplets/%s/frac%s_vs_pt 'Fraction of TPs with %s %s vs p_{T}; TP transverse momentum p_{T} [GeV]; Fraction' SimNtuplets/%s/num_pt_%s general/num_vs_pt" % strings)
        output.append("SimNtuplets/%s/frac%s_vs_eta 'Fraction of TPs with %s %s vs #eta; TP pseudorapidity #eta; Fraction' SimNtuplets/%s/num_eta_%s general/num_vs_eta" % strings)
        output.append("SimNtuplets/%s/frac%s_vs_vertpos 'Fraction of TPs with %s %s vs r_{vertex}; TP radial vertex position wrt beamspot r_{vertex} [cm]; Fraction' SimNtuplets/%s/num_vertpos_%s general/num_vs_vertpos" % strings)
    return output

SimNtupletHistograms = buildSimNtupletFractions("longest", "longest SimNtuplet") + buildSimNtupletFractions("mostAlive", "most alive SimNtuplet")

postProcessorSimNtuplets = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsTop),
    efficiency = cms.vstring(
        SimNtupletHistograms
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