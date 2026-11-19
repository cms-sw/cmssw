import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# =============================================================================
# Helper utilities
# =============================================================================

def _addNoFlow(module):
    """Add noFlowDists entries for all histograms referenced in efficiency
    strings, mirroring the track post processor convention."""
    _noflowSeen = set()
    for eff in module.efficiency.value():
        tmp = eff.split(" ")
        if "cut" in tmp[0]:
            continue
        ind = -1
        if tmp[ind] == "fake" or tmp[ind] == "simpleratio":
            ind = -2
        if tmp[ind] not in _noflowSeen:
            module.noFlowDists.append(tmp[ind])
            _noflowSeen.add(tmp[ind])
        if tmp[ind - 1] not in _noflowSeen:
            module.noFlowDists.append(tmp[ind - 1])
            _noflowSeen.add(tmp[ind - 1])


def makeSVEfficiencyBundle(histoSuffix, label):
    """Produce the list of efficiency/rate strings for a given SV kinematic
    variable, following the DQMGenericClient string format:

      outputHistoName 'Plot title' numeratorHisto denominatorHisto [fake]

    The histogram naming convention matches SVMonitoringBundle:
      num_sim_<suffix>
      num_assoc(simToReco)_<suffix>
      num_reconstructableSim_<suffix>
      num_assoc(reconstructableSimToReco)_<suffix>
      num_merged_<suffix>
      num_reco_<suffix>
      num_assoc(recoToSim)_<suffix>
      num_duplicate_<suffix>
      num_fake_<suffix>          (no sim match at all)
      num_pileup_<suffix>        (matched to pileup sim SV)

    Args:
        histoSuffix (str): suffix used in histogram names, e.g. 'decayLength'
        label (str):       ROOT label for the x-axis, e.g. 'L_{3D} [cm]'
    """
    s = histoSuffix

    return [
        # --- Efficiency ---
        "effic_vs_{s} 'Efficiency vs {l} (Sim)' "
        "num_assoc(simToReco)_{s} num_sim_{s}".format(s=s, l=label),

        # --- Technical efficiency (reconstructable sim SVs only) ---
        "techEffic_vs_{s} 'Technical efficiency vs {l} (Sim)' "
        "num_assoc(reconstructableSimToReco)_{s} "
        "num_reconstructableSim_{s}".format(s=s, l=label),

        # --- Merge rate (sim SV merged with another sim SV into one reco SV) ---
        "mergeRate_vs_{s} 'Merge rate vs {l} (Sim)' "
        "num_merged_{s} num_sim_{s}".format(s=s, l=label),

        # --- Fake rate (reco SV with no sim match) ---
        "fakeRate_vs_{s} 'Fake rate vs {l} (Reco)' "
        "num_assoc(recoToSim)_{s} num_reco_{s} fake".format(s=s, l=label),

        # --- Duplicate rate (reco SV matched to already-claimed sim SV) ---
        "duplicateRate_vs_{s} 'Duplicate rate vs {l} (Reco)' "
        "num_duplicate_{s} num_reco_{s}".format(s=s, l=label),

        # --- Pileup rate (reco SV matched only to pileup sim SVs) ---
        "pileupRate_vs_{s} 'Pileup rate vs {l} (Reco)' "
        "num_pileup_{s} num_reco_{s}".format(s=s, l=label),
    ]


def makeSVPerPdgEfficiencyBundle(histoSuffix, label):
    """Produce per-PDG efficiency strings for b-hadron, c-hadron, and other
    origin sim SVs. These use the per-PDG histograms booked when
    doPerPdgPlots=True in the analyzer.

    Histogram naming: num_sim_b_<suffix>, num_assoc(simToReco)_b_<suffix>, etc.
    """
    s = histoSuffix
    result = []
    for origin, tag in [("B-hadron", "b"), ("D-hadron", "c"), ("other", "other")]:
        result += [
            "effic_{t}_vs_{s} 'Efficiency ({o} origin) vs {l} (Sim)' "
            "num_assoc(simToReco)_{t}_{s} "
            "num_sim_{t}_{s}".format(s=s, l=label, o=origin, t=tag),

            "techEffic_{t}_vs_{s} 'Technical efficiency ({o} origin) vs {l} (Sim)' "
            "num_assoc(reconstructableSimToReco)_{t}_{s} "
            "num_reconstructableSim_{t}_{s}".format(s=s, l=label, o=origin, t=tag),
        ]
    return result


# =============================================================================
# Subdirectory configuration
# =============================================================================

# Default subdirs: one entry per reco SV collection, wildcarded.
# The analyzer books histograms under rootFolder/collectionLabel/.
# Adjust these to match the collections configured in the analyzer cfi.
_defaultSVSubdirs = [
    "Validation/Vertices/Secondary/*",
]

_defaultSVSubdirsSummary = [d.replace("/*", "") for d in _defaultSVSubdirs]

# =============================================================================
# Main post processor
# =============================================================================

postProcessorSecondaryVertex = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSVSubdirs),
    efficiency = cms.vstring(
        # Decay length — the most SV-specific and b-tagging-relevant quantity
        makeSVEfficiencyBundle("decayLength",    "L_{3D} [cm]") +
        makeSVEfficiencyBundle("decayLengthSig", "L_{3D}/#sigma_{L_{3D}}") +
        makeSVEfficiencyBundle("decayLengthXY",  "L_{2D} [cm]") +

        # Track multiplicity
        makeSVEfficiencyBundle("nTracks", "N tracks at SV") +

        # Kinematics
        makeSVEfficiencyBundle("eta",  "#eta") +
        makeSVEfficiencyBundle("phi",  "#phi") +
        makeSVEfficiencyBundle("pt",   "p_{T}") +
        makeSVEfficiencyBundle("mass", "SV invariant mass [GeV]") +

        # Fit quality
        makeSVEfficiencyBundle("chi2ndof", "Normalised #chi^{2}") +

        []  # placeholder for future additions
    ),
    resolution = cms.vstring(
        # Position resolution profiles — mean and RMS extracted by
        # DQMGenericClient from the 2D residual histograms.
        "x_res_vs_decayLength 'x resolution vs L_{3D};Simulated L_{3D} [cm];#sigma(x) [cm]' x_res_vs_decayLength",
        "y_res_vs_decayLength 'y resolution vs L_{3D};Simulated L_{3D} [cm];#sigma(y) [cm]' y_res_vs_decayLength",
        "z_res_vs_decayLength 'z resolution vs L_{3D};Simulated L_{3D} [cm];#sigma(z) [cm]' z_res_vs_decayLength",
        "x_res_vs_decayLengthXY 'x resolution vs L_{2D};Simulated L_{2D} [cm];#sigma(x) [cm]' x_res_vs_decayLengthXY",
        "y_res_vs_decayLengthXY 'y resolution vs L_{2D};Simulated L_{2D} [cm];#sigma(y) [cm]' y_res_vs_decayLengthXY",
        "z_res_vs_decayLengthXY 'z resolution vs L_{2D};Simulated L_{2D} [cm];#sigma(z) [cm]' z_res_vs_decayLengthXY",
        "x_res_vs_nTracks 'x resolution vs N tracks;Number of tracks in RecoSV;#sigma(x) [cm]' x_res_vs_nTracks",
        "y_res_vs_nTracks 'y resolution vs N tracks;Number of tracks in RecoSV;#sigma(y) [cm]' y_res_vs_nTracks",
        "z_res_vs_nTracks 'z resolution vs N tracks;Number of tracks in RecoSV;#sigma(z) [cm]' z_res_vs_nTracks",
        # Decay length resolution
        "decayLength_res_vs_decayLength 'L_{3D} resolution vs L_{3D};Simulated L_{3D} [cm];#sigma(L_{3D}) [cm]' decayLength_res_vs_decayLength",
        "decayLength_res_vs_decayLengthXY 'L_{3D} resolution vs L_{2D};Simulated L_{2D} [cm];#sigma(L_{3D}) [cm]' decayLength_res_vs_decayLengthXY",
        "decayLength_res_vs_nTracks 'L_{3D} resolution vs N tracks;Number of tracks in RecoSV;#sigma(L_{3D}) [cm]' decayLength_res_vs_nTracks",
        "decayLength_res_vs_eta 'L_{3D} resolution vs #eta;Simulated #eta;#sigma(L_{3D}) [cm]' decayLength_res_vs_eta",
        "decayLengthXY_res_vs_decayLength 'L_{2D} resolution vs L_{3D};Simulated L_{3D} [cm];#sigma(L_{2D}) [cm]' decayLengthXY_res_vs_decayLength",
        "decayLengthXY_res_vs_decayLengthXY 'L_{2D} resolution vs L_{2D};Simulated L_{2D} [cm];#sigma(L_{2D}) [cm]' decayLengthXY_res_vs_decayLengthXY",
        "decayLengthXY_res_vs_nTracks 'L_{2D} resolution vs N tracks;Number of tracks in RecoSV;#sigma(L_{2D}) [cm]' decayLengthXY_res_vs_nTracks",
        "decayLengthXY_res_vs_eta 'L_{2D} resolution vs #eta;Simulated #eta;#sigma(L_{2D}) [cm]' decayLengthXY_res_vs_eta",
        # Kinematics resolution
        "eta_res_vs_decayLength '#eta resolution vs L_{3D};L_{3D} [cm];#sigma(#eta)' eta_res_vs_decayLength",
        "eta_res_vs_decayLengthXY '#eta resolution vs L_{2D};L_{2D} [cm];#sigma(#eta)' eta_res_vs_decayLengthXY",
        "eta_res_vs_nTracks '#eta resolution vs N tracks;Number of tracks in RecoSV;#sigma(#eta)' eta_res_vs_nTracks",
        "phi_res_vs_decayLength '#phi resolution vs L_{3D};L_{3D} [cm];#sigma(#phi) [rad]' phi_res_vs_decayLength",
        "phi_res_vs_decayLengthXY '#phi resolution vs L_{2D};L_{2D} [cm];#sigma(#phi) [rad]' phi_res_vs_decayLengthXY",
        "phi_res_vs_nTracks '#phi resolution vs N tracks;Number of tracks in RecoSV;#sigma(#phi) [rad]' phi_res_vs_nTracks",
        "pt_res_vs_decayLength 'p_{T} resolution vs L_{3D};L_{3D} [cm];#sigma(p_{T}) [GeV]' pt_res_vs_decayLength",
        "pt_res_vs_decayLengthXY 'p_{T} resolution vs L_{2D};L_{2D} [cm];#sigma(p_{T}) [GeV]' pt_res_vs_decayLengthXY",
        "pt_res_vs_nTracks 'p_{T} resolution vs N tracks;Number of tracks in RecoSV;#sigma(p_{T}) [GeV]' pt_res_vs_nTracks",
        "mass_res_vs_decayLength 'Mass resolution vs L_{3D};L_{3D} [cm];#sigma(m) [GeV]' mass_res_vs_decayLength",
        "mass_res_vs_decayLengthXY 'Mass resolution vs L_{2D};L_{2D} [cm];#sigma(m) [GeV]' mass_res_vs_decayLengthXY",
        "mass_res_vs_nTracks 'Mass resolution vs N tracks;Number of tracks in RecoSV;#sigma(m)' mass_res_vs_nTracks",
    ),
    noFlowDists = cms.untracked.vstring(
        "trackPurity",
        "trackEfficiency",
    ),
)
_addNoFlow(postProcessorSecondaryVertex)

# =============================================================================
# Per-PDG post processor (b/c/other efficiency breakdowns)
# =============================================================================

postProcessorSecondaryVertexPerPdg = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSVSubdirs),
    efficiency = cms.vstring(
        makeSVPerPdgEfficiencyBundle("decayLength",    "L_{3D} [cm]") +
        makeSVPerPdgEfficiencyBundle("decayLengthSig", "L_{3D}/#sigma_{L_{3D}}") +
        makeSVPerPdgEfficiencyBundle("r",              "r_{T} [cm]") +
        makeSVPerPdgEfficiencyBundle("nTracks",        "N tracks at SV") +
        makeSVPerPdgEfficiencyBundle("eta",            "#eta") +
        makeSVPerPdgEfficiencyBundle("mass",           "SV invariant mass [GeV]") +
        []
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
)
_addNoFlow(postProcessorSecondaryVertexPerPdg)

# =============================================================================
# Summary post processor (one bin per collection for cross-collection comparison)
# =============================================================================

postProcessorSecondaryVertexSummary = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSVSubdirsSummary),
    efficiency = cms.vstring(
        # One bin per collection — histograms booked with collection label on x-axis
        "effic_vs_coll 'Efficiency vs SV collection' "
        "num_assoc(simToReco)_coll num_sim_coll",

        "techEffic_vs_coll 'Technical efficiency vs SV collection' "
        "num_assoc(reconstructableSimToReco)_coll num_reconstructableSim_coll",

        "fakeRate_vs_coll 'Fake rate vs SV collection' "
        "num_assoc(recoToSim)_coll num_reco_coll fake",

        "duplicateRate_vs_coll 'Duplicate rate vs SV collection' "
        "num_duplicate_coll num_reco_coll",

        "mergeRate_vs_coll 'Merge rate vs SV collection' "
        "num_merged_coll num_reco_coll",

        "pileupRate_vs_coll 'Pileup rate vs SV collection' "
        "num_pileup_coll num_reco_coll",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
)
_addNoFlow(postProcessorSecondaryVertexSummary)

# =============================================================================
# Sequence
# =============================================================================

postProcessorSecondaryVertexSequence = cms.Sequence(
    postProcessorSecondaryVertex +
    postProcessorSecondaryVertexPerPdg +
    postProcessorSecondaryVertexSummary
)
