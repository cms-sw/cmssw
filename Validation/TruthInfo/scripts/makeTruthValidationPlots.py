#!/usr/bin/env python3
"""Render the truth-branch DQM output as a CMS-styled, browsable gallery.

Each variable becomes ONE overlay plot with a ratio panel, rather than isolated
histograms the reader has to compare by eye. The reco-driven metrics (fake, pileup,
purity, resolution) overlay the branch-association working points; the truth-driven
metrics (efficiency, duplicate, split, composition) overlay the graph LEVELS, because
their folders are keyed by level and the working point never enters them.

Collections, working points, levels and categories are DISCOVERED from the DQM folder
names, so a new collection, working point or level needs no edit here.

  makeTruthValidationPlots.py DQM_V0001_*.root --outputDir plots
"""
import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # batch backend, must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import ROOT  # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

plt.style.use(hep.style.CMS)

# The reference working point: every reco-driven ratio is taken against it, and it
# keeps the first colour of the Petroff cycle everywhere in the gallery so colour
# follows the entity.
REFERENCE_WP = "Fixed"
WP_ORDER = ["Fixed", "AdaptiveTight", "AdaptiveNominal", "AdaptiveLoose"]
# Truth-driven folders are keyed by graph LEVEL (hit-based domains) or by the vertex
# resolution (composite domains). Every truth-driven ratio is taken against
# caloBoundary, falling back to the first suffix present. The signal suffix is the
# overall signal entry: its denominator is the preset SEED objects among the selected
# roots, so with a selection preset it is the efficiency of the signal object itself
# (the tau, not its decay legs). signalNoSelection is the same seed objects with no
# selector cut at all, so the efficiency is quoted against every seed in the event.
# Every entry here is an ANTICHAIN: no member is an ancestor of another, so no efficiency
# counts one object twice. That is the entry requirement for a truth denominator.
LEVEL_ORDER = ["stableLegsFromUpstream", "caloBoundary", "stableDecayProducts", "hardProcess",
               "reconstructableFromSignal", "reconstructableFinalState", "underlyingEvent",
               "partonJets", "bHadrons", "cHadrons", "visibleTau", "signal", "signalNoSelection"]
# What each truth-driven series IS. These are the efficiency DENOMINATORS, and they are not
# interchangeable, but every one of them is an ANTICHAIN: no member is an ancestor of
# another, so no efficiency counts one object twice. Sizes quoted are ttbar PU200 D122 with
# the top preset, per event, measured from the associator target lists.
LEVEL_MEANING = {
    "caloBoundary":
        "every particle recorded crossing the tracker/calorimeter boundary OUTWARD, back-scattered tracks "
        "excluded. This is what actually arrived at the calorimeter, including secondaries created in the "
        "tracker, so it is the natural denominator for a CALORIMETRIC collection. About 80 per event, of which "
        "74% are also generator-stable.",
    "stableDecayProducts":
        "the generator's final state: GEN particles with status 1. Defined by the generator alone, independent "
        "of any detector or selection, so it includes neutrinos and everything that never reaches a "
        "calorimeter. About 2740 per event, of which only 2.2% reach the calorimeter boundary: as a "
        "calorimetric denominator it is therefore dominated by objects no calorimeter object could ever match.",
    "stableLegsFromUpstream":
        "the LEAVES of the selected subgraph: follow every particle out of the artificial Upstream vertex down "
        "until the chain stops. It is the selection's own notion of 'the interesting activity', so it exists "
        "only when a selection preset ran. About 35 per event; 67% of it is also at the calorimeter boundary "
        "and 44% is also generator-stable, so it is a middle ground between the two.",
    "hardProcess":
        "the OUTGOING LEGS of the hard scatter, and NOT the resonance despite the name. isHardProcess is set on "
        "the hard-scatter participants and the deepest-element antichain keeps the outgoing ones, so on ttbar "
        "this holds b, b~ and the W decay products rather than the two tops; on H to two photons it holds the "
        "photons; on VBF the two tagging quarks plus the four neutrinos. Measured on one event of each of the "
        "eleven generator templates. Use SIGNAL for the resonance. Empty for a particle gun, which has no "
        "hard-process record at all.",
    "reconstructableFromSignal":
        "the VISIBLE FINAL STATE of the resonance: walk down from each signal root and stop at the first thing a "
        "detector reconstructs as an object. That is not the first generator-stable particle. A pi0 decays to two "
        "photons at once, but the analysis reconstructs the pi0, so the pi0 is the entry and its photons are not; "
        "a three-prong tau contributes its three charged pions. Intermediate resonances the detector never sees "
        "as objects (a1, rho) are walked through, unless their pdg id is added to reconstructablePdgIds on the "
        "graph. Neutrinos are dropped, being invisible, which is why an all-neutrino final state gives an EMPTY "
        "level rather than a wrong one. An antichain by construction, since the walk stops at each leg. "
        "TenTau no-PU: 18.59 per event against 10.00 signal objects. This is the denominator to read for "
        "'was what the resonance actually produced reconstructed'. EMPTY on a sample produced before the signal "
        "flag existed, since it is stamped at DIGI.",
    "underlyingEvent":
        "the stable legs of the UNDERLYING EVENT, the spectator activity hanging off the artificial "
        "UnderlyingEvent vertex. The counterpart of stableLegsFromUpstream, which holds the ISR and upstream "
        "side of the same interaction, so between them and the signal levels the event is partitioned into what "
        "the analysis asked for, what radiated into it, and what came along with it. An antichain: a leg is a "
        "particle that produced nothing further. Exists only when a selection preset ran, since the artificial "
        "vertices are what the preset builds, and is EMPTY rather than wrong otherwise.",
    "signal":
        "the preset's seed species among the selected roots, that is THE RESONANCE itself and not its decay "
        "products: two tops for the top preset, one Z for Drell-Yan, one Higgs for VBF and ggF, ten taus for "
        "the TenTau gun. Verified present in all eleven generator templates. This is the series that answers "
        "'was the object I generated reconstructed', and the one to read when hardProcess looks wrong.",
    "signalNoSelection":
        "the same seed species with the kinematic selector removed. Equal to signal whenever every seed passes "
        "the selector, which is the case for tops; a difference between the two is the selector's own cost.",
}

VERTEX_SUFFIXES = ["interaction", "immediate"]
TRUTH_SUFFIXES = LEVEL_ORDER + VERTEX_SUFFIXES
REFERENCE_LEVEL = "caloBoundary"
# The metrics that live in the level-keyed folders; everything else is per working point.
TRUTH_METRICS = {"composition", "efficiency", "duplicate", "splitrate"}

# Which metrics we plot, and how each should be read.
METRICS = {
    "composition": (
        "Branch composition",
        "What the selected truth objects ARE, as fractions of the efficiency denominator, split by the Geant4 "
        "process that created them. Read this page first: it says what the other pages are averaging over.",
        "num_simul_reason / sum(num_simul_reason)",
    ),
    "efficiency": (
        "Branch efficiency (TruthToReco)",
        "Of the truth objects at one graph level, the fraction reconstructed AS ONE OBJECT: some single reco "
        "object covered enough of it and was not mostly something else. The truth target is fixed a priori by "
        "the level, so the reco-driven working point never enters; the curves compare the levels themselves. "
        "Each level is drawn as a PAIR: individual (filled, solid) counts a truth object as found when a "
        "single reco object covers it, cumulative (open, dashed, same colour) when all reco objects of the "
        "collection together cover it. A multi-prong decay separates the two: three pions each with their own "
        "trackster leave the tau individually lost but cumulatively found.",
        "num_assoc(simToReco) / num_simul, cumulative: num_assoc_cumulative / num_simul",
    ),
    "duplicate": (
        "Duplicate rate (TruthToReco)",
        "Of the selected truth objects, the fraction that MORE THAN ONE reco object individually reconstructed. "
        "Redundant reconstruction, distinct from splitting. NOT BOOKED FOR CALORIMETRIC COLLECTIONS, so this "
        "page has tracking folders only: the criterion asks for more than one reco object each missing less than "
        "20% of the SAME branch energy, and two objects built from disjoint layer clusters have scores summing to "
        "at least one, so it cannot fire. Every TICL collection validated here partitions its layer clusters "
        "(measured on 200 no-PU ttbar events: no layer cluster is used by two tracksters of one collection). "
        "For energy the meaningful counterpart of reconstructing something twice is the SPLIT page.",
        "num_duplicate / num_simul",
    ),
    "splitrate": (
        "Split rate (TruthToReco)",
        "Of the selected truth objects, the fraction that no single reco object reconstructed but that several "
        "together cover. This is the collective case: the truth object's subgraph was reconstructed in pieces. "
        "Efficiency, duplicate and split are mutually exclusive, so together with lost they sum to one.",
        "num_split / num_simul",
    ),
    "recopurity": (
        "Reco purity (RecoToTruth)",
        "Of a reco object, the fraction that belongs to the truth object it matched. The RECO object is the "
        "denominator, which is what distinguishes it from truth purity. This is where the adaptive working "
        "points earn their keep: the climb stops at the graph level that matches the object, so purity rises "
        "sharply while the lost fraction is unchanged.",
        "num_recopurity / num_reco",
    ),
    "fakerate": (
        "Fake rate (RecoToTruth)",
        "Of the reco objects, the fraction NO truth branch owns: its hits come from several different generated "
        "particles with none dominating, so there is nothing to attribute it to. Dominance is measured over an "
        "ANTICHAIN of the graph, the level named by dominanceLevel, and that restriction is what makes the "
        "question well posed: over a set containing both a particle and its descendants the leader and the "
        "runner-up can be the same particle at two depths, which read as 'nothing dominates' on no-PU TenTau "
        "where ten isolated taus must each give one overwhelming winner (leading share 0.26 unrestricted, 0.98 "
        "over caloBoundary). An object counts as owned when one branch of that level carries at least "
        "minLeadingTruthShare = 0.5 of the shared quantity all of them contribute. An object matched to NOTHING "
        "is a fake too, and the NOCANDIDATE page is that failure mode on its own. An object that matched truth "
        "but has no candidate at the dominance level is NOT a fake: the question is undefined for it, and "
        "counting it would measure how much of the event the level covers rather than how well the collection "
        "reconstructs. That category has its own NOLEVELCANDIDATE page. "
        "A fraction of OBJECTS, one per object. It is not one minus the purity: an object can be owned and "
        "impure, and the purity page is where that shows. Identical at all four working points by construction, "
        "since dominance is read from the only map carrying every candidate; the climb changes which branch an "
        "object is attributed to, never whether one dominates.",
        "1 - num_dominated / num_reco",
    ),
    "nocandidate": (
        "No-candidate rate (RecoToTruth)",
        "Of the reco objects, the fraction matched to NOTHING in the truth graph. This is one of the two ways of "
        "being a fake and is a subset of the fake page, not a competing definition of it: the other way is having "
        "candidates with no dominant one. Read the two together to see which mechanism a collection suffers from. "
        "At PU200 this one saturates near zero, because the graph is dense enough that almost every reco object "
        "overlaps some branch, so it stops discriminating and the fake and purity pages carry the information. "
        "A composite domain matches everything it builds, so it is near zero there by construction too.",
        "1 - num_assoc(recoToSim) / num_reco",
    ),
    "contaminated": (
        "Contaminated rate (RecoToTruth), calorimetry only",
        "Of the reco objects, the fraction whose best truth candidate does NOT pass HGCalValidator's non-fake "
        "criterion, recoToSim score below maxRecoToSimScore = 0.6. This is NOT a fake rate and the fake page is "
        "the one to read for that. The score is normalised against the cell's TOTAL truth energy, so a cell "
        "shared with overlaid interactions pushes it towards 1 even for a perfectly matched object, and at PU200 "
        "it saturates: measured on ttbar PU200, ticlCandidate AdaptiveNominal, 73.8% of tracksters fail this cut "
        "while only 2.3% have no truth candidate at all. Read it as cell-level contamination, which is what the "
        "reconstruction is up against, and use it to compare against HGCalValidator, which applies the same cut.",
        "1 - num_assoc_strict / num_reco",
    ),
    "nolevelcandidate": (
        "No dominance-level candidate rate (RecoToTruth)",
        "Of the reco objects, the fraction that matched truth but whose candidates include nothing at the "
        "dominance level, so the fake question is UNDEFINED for them rather than answered. Read it as coverage "
        "of the level, not as a reconstruction failure: on no-PU ttbar it is 32.5% of tracksters and 36.8% of "
        "tracks, while only 0.3% of tracks match nothing at all, and moving the tracking level to "
        "stableDecayProducts only takes it to 27.7%. A large value here means the fake rate is being formed on "
        "a small subset of the collection and should be quoted with this page beside it.",
        "1 - num_levelcandidate / num_reco",
    ),
    "pileuprate": (
        "Pileup rate (RecoToTruth)",
        "Of the reco objects, the fraction whose match belongs to a pileup interaction rather than the signal "
        "one. The graph answers this directly because every particle and vertex carries its interaction id.",
        "num_pileup / num_reco",
    ),
    "resolution": (
        "Residual mean and width",
        "Gaussian fit, slice by slice, of (reco - truth)/truth for the momentum and of (reco - truth) for the "
        "angles, against the TRUTH variable. Mean is the bias, Sigma the width. Read these with the reference in "
        "mind: it is the BRANCH ROOT, and a reco object matched to a branch by shared hits can correspond to a "
        "DESCENDANT of that root, so the width is dominated by the branch definition, not by tracking. It is "
        "therefore a diagnostic of the truth definition and not a tracking resolution. Slices with fewer than "
        "20 entries, and fits whose width exceeds the fit range or collapses below one bin, are not drawn.",
        "Gaussian slice fit of ptres_vs_*, etares_vs_*, phires_vs_*",
    ),
}
VARIABLE_MEANING = {
    "pt": "branch root transverse momentum",
    "eta": "pseudorapidity of the branch ROOT, that is where it was produced, NOT where its energy landed; only "
           "for the caloBoundary level are the two the same",
    "caloeta": "pseudorapidity at which the branch ENTERS the calorimeter, taken from the boundary crossing of its "
               "most energetic particle to reach it; a branch that never reached the calorimeter is not plotted, so "
               "this axis is restricted to what a calorimeter could have seen",
    "phi": "branch root azimuth",
    "nhits": "hits of the branch footprint in the truth hit index",
    "vertpos": "radius of the branch production vertex",
    "zpos": "z of the branch production vertex",
    "dxy": "transverse impact parameter of the branch",
    "dz": "longitudinal impact parameter of the branch",
    "depth": "number of ancestors of the branch root in the graph, that is how far down the event history it sits",
    "root_footprint_fraction": "fraction of the branch tracker footprint that belongs to the root particle itself rather than to "
                "its descendants; near 1 is a clean single particle",
    "flavour": "species that initiated the truth object, read off the branch root PDG id. Only the partonJets level has parton roots, so on every other level the whole distribution sits in the `other` bin by construction",
    "shared_energy_fraction": "fraction of the truth branch energy that the matched reco object shares with it",
}
# Axis title per variable, in the CMS convention: the unit in square brackets, and no
# bracket at all for a pure count, a fraction or a dimensionless shape variable.
# Bin names of the flavour axis, mirroring truth::kFlavourBinNames. The axis is species
# rather than a number, so the ticks are named and never drawn as bin indices.
FLAVOUR_BINS = ["other", "d", "u", "s", "c", "b", "t", "g"]

# Axes booked with symlog bin edges, mirroring _linthresh in truthBranchValidation_cff.
# The value is the linear-to-log crossover, and it must match the booking or the drawn
# axis and the bin edges disagree.
# (linthresh, max). The maximum is needed because matplotlib's symlog is SYMMETRIC about
# zero by default: without an explicit left limit it draws a negative branch, and a
# negative pt or radius cannot exist, so half the canvas went to empty mirror decades.
SYMLOG_AXES = {
    "pt": (0.1, 1000.0),
    "vertpos": (0.001, 60.0),
    "nhits": (1.0, 10000.0),
}

AXIS_TITLE = {
    "flavour": "initiating parton",
    "pt": r"p$_{T}$ [GeV]",
    # The truth object here is the branch ROOT, whose eta is where it was PRODUCED, not
    # where its energy landed. Say so on the axis: for anything but caloBoundary the root
    # decayed long before the calorimeter, so this is not an acceptance axis. The
    # calorimeter-entrance counterpart is caloeta.
    "eta": r"truth root $\eta$",
    "phi": r"$\phi$ [rad]",
    "nhits": "number of hits",
    "vertpos": "vertex radius [cm]",
    "zpos": "vertex z [cm]",
    "dxy": r"d$_{xy}$ [cm]",
    "dz": r"d$_{z}$ [cm]",
    "depth": "depth (number of ancestors)",
    "root_footprint_fraction": "root footprint fraction",
    "caloeta": r"$\eta$ at calorimeter entrance",
    "shared_energy_fraction": "shared energy fraction",
    "reason": "creation process",
}
# Axes drawn over their full booked range whatever the sample populates. A gun sample
# fills a slice of eta and autoscaling would hide that the rest of the acceptance is
# empty, which is itself the result.
# The eta axes are booked over +-4.5, so the drawing range matches the booking; a
# narrower window drops the forward region off the canvas.
AXIS_RANGE = {"eta": (-4.5, 4.5), "caloeta": (-4.5, 4.5)}
# Residual axis titles per fitted quantity: the momentum residual is relative and so
# dimensionless, the angular ones are differences and phi carries radians.
RESIDUAL_TITLE = {
    "pt": "(reco - truth) / truth",
    "eta": "reco - truth",
    "phi": "reco - truth [rad]",
}
RESIDUAL_UNIT = {"pt": "", "eta": "", "phi": " [rad]"}
# One marker shape and one line style per series, so the curves stay separable in
# greyscale and under colour-vision deficiency, not by colour alone. A cumulative
# partner keeps its series' colour and shape and is drawn open and dashed.
SERIES_MARKERS = ["o", "s", "^", "v", "*", "P", "D"]
SERIES_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 1))]
# Typography. The CMS style is built for a single full-page pad, where a 26 pt axis
# title is right; on a two-pad figure carrying a twelve-entry legend it dwarfs
# everything around it and a long y title outgrows its own pad. The title is therefore
# modestly larger than the tick labels, and the ratio pad's title smaller still so it
# does not compete with the main pad's.
AXIS_TITLE_SIZE = 17
TICK_LABEL_SIZE = 15
RATIO_TITLE_SIZE = 12


def axis_title(var):
    """The axis title of a variable, with its unit where it has one."""
    return AXIS_TITLE.get(var, var)


def marker_size(marker):
    """A star needs more area than a circle to read as the same size."""
    return 8 if marker == "*" else 5
# The Individual-match criterion per category: (legend line, full statement). The
# threshold values come from the corresponding standard validation, not from here; the
# full statement says where each lives. The legend line goes on every truth-driven
# plot, the full statement into the DEFINITIONS text, so the number on the page
# carries its own definition.
_VERTEX_CRITERION = (
    "Individual: any positive shared p$_{T}^{2}$ track fraction (vertex validation standard)",
    "Individual: any positive shared pt^2 track fraction. The reference vertex association gates on "
    "POSITION and ships its shared-track-fraction cut disabled, sharedTrackFraction = -1.0 "
    "(SimTracker/VertexAssociation/plugins/VertexAssociatorByPositionAndTracksProducer.cc:72, the "
    "fraction branch at src/VertexAssociatorByPositionAndTracks.cc:129), so on the shared-components "
    "axis used here the reference criterion is any positive shared fraction.",
)
MATCH_CRITERIA = {
    "Tracking": (
        "Individual: track shares > 75% of its own hits with the branch (MTV standard)",
        "Individual: a track shares more than 75% of its own hits with the branch, with no "
        "truth-normalised cut. That is the QuickTrackAssociatorByHits criterion MultiTrackValidator "
        "counts efficiency with: Cut_RecoToSim = 0.75, Purity_SimToReco = 0.75, Quality_SimToReco = 0.5 "
        "with SimToRecoDenominator = 'reco', so every cut acts on the reco-normalised fraction "
        "(SimTracker/TrackAssociatorProducers/python/quickTrackAssociatorByHits_cfi.py:4-8, applied in "
        "plugins/QuickTrackAssociatorByHitsImpl.cc:234-244 and 312-326; MultiTrackValidator adds no "
        "further cut, plugins/MultiTrackValidator.cc:939-943).",
    ),
    "Vertexing": _VERTEX_CRITERION,
    "SecondaryVertexing": _VERTEX_CRITERION,
    "Calorimetry": (
        "Individual: shared energy fraction > 0.5 of the branch energy in this collection's detectors "
        "(HGCal standard)",
        "Individual: a single trackster shares more than 0.5 of the truth branch's energy IN THE DETECTORS "
        "THIS COLLECTION RECONSTRUCTS, and its own "
        "recoToSim score is below 0.6. The denominator is not the branch's whole calorimetric energy: the "
        "truth graph keeps every calorimeter deposit of the branch, barrel included, and their sampling "
        "energies differ by orders of magnitude, so on 200 no-PU ttbar events only 0.5% to 10% of a top "
        "branch's calorimetric energy is in HGCAL and no trackster could reach half of the whole. The "
        "reference quantity has no such problem because a sim trackster exists only in HGCAL. "
        "These are three DIFFERENT axes in HGCalValidator, not one: "
        "efficiency is a SHARED ENERGY FRACTION cut, minTSTSharedEneFracEfficiency = 0.5 "
        "(Validation/HGCalValidation/python/HGVHistoProducerAlgoBlock_cfi.py:82, applied in "
        "src/HGVHistoProducerAlgo.cc:2897); purity and duplicate cut the simToReco SCORE below 0.2 "
        "(maxSimToRecoScoreForPurity/Duplicate, cfi:72-73, applied at :2898-2899); fake and merge cut "
        "the recoToSim score below 0.6 (maxRecoToSimScoreForNonFake/Merge, cfi:70-71, applied at "
        ":2819-2820). Both scores are the TICL ones, computed here exactly as "
        "SimCalorimetry/HGCalAssociatorProducers/plugins/"
        "AllTracksterToSimTracksterAssociatorsByHitsProducer.cc:341-364 and :428-453 do: the squared "
        "energy the other side fails to cover, over the squared self energy, with an excess on the "
        "other side counting as a good association.",
    ),
}
# Per-domain caveats. A number that is correct but not discriminating reads as a result
# unless the page says otherwise, so the page says otherwise.
CATEGORY_NOTE = {
    "Vertexing": (
        "A vertex owns no hits, so it is associated to a truth vertex by aggregating the tracks it was built "
        "from, weighted by pt SQUARED. That is the weighting CMSSW's own vertex association uses: "
        "calculateVertexSharedTracks returns sharedPt2Fraction as sum(pt^2 of shared tracks) over sum(pt^2 of "
        "ALL the vertex's tracks). Each track carries its best-matched particle, and that particle is counted at "
        "its INTERACTION, so a track from a decay downstream of the vertex still belongs to the interaction its "
        "chain started from. Purity is then the leading interaction's share of the vertex pt^2. Measured on "
        "ttbar with no pileup it is 0.973 with efficiency 1.000; under PU200 it drops to 0.029, which is far "
        "below what vertex reconstruction can plausibly be doing and is not yet understood. Treat the pileup "
        "number as an open question, not as a result."
    ),
    "SecondaryVertexing": (
        "Same aggregation as Vertexing, and it is the case the immediate-production-vertex definition suits best: "
        "a secondary vertex IS a decay or interaction vertex, so the tracks that belong to it were produced there."
    ),
    "Calorimetry": (
        "Tracksters are matched on SHARED ENERGY in the calorimeter channel, the same quantity the TICL trackster "
        "validation scores against. One thing differs from TICL and is worth knowing: TICL weights each cell by its "
        "RECHIT energy, while a trackster reaches the truth graph as (DetId, fraction) with no per-cell reco energy, "
        "so the weight here is the cell's total energy in the truth hit index. The truth denominator spans the whole "
        "selector acceptance, so the efficiency correctly falls to zero outside the HGCAL coverage rather than being "
        "renormalised to it."
    ),
}
STYLE = (
    "body{font-family:sans-serif;margin:2em;max-width:1500px;color:#222}"
    "h1{margin-bottom:.2em}h2{border-bottom:1px solid #ccc;padding-bottom:.2em;margin-top:1.6em}"
    "img{border:1px solid #ddd;margin:4px;vertical-align:top}"
    "a{color:#036;text-decoration:none}a:hover{text-decoration:underline}"
    ".grid{display:flex;flex-wrap:wrap}"
    ".def{background:#f4f6f8;border-left:4px solid #5790fc;padding:.8em 1em;margin:1em 0;max-width:62em}"
    ".f{font-family:monospace;background:#fff;padding:.2em .45em;border:1px solid #dde}"
    "ul.idx{line-height:1.8;max-width:62em}"
)
# Plots the graph makes possible that a frozen truth object cannot answer. Shown in the
# gallery so the next step is visible rather than tribal knowledge.
PROPOSED = [
    ("Merge rate by lowest common ancestor",
     "When two branches are reconstructed as one object the graph gives the LCA of the contributors, so the merge "
     "rate can be plotted against the LCA pdgId: WHICH physical object the merge corresponds to, for instance a "
     "pi0 whose two photons merged, not merely that a merge happened."),
    ("Adaptive-level agreement",
     "Fraction of reco objects whose adaptive level equals the fixed best match, versus pt and eta. The direct "
     "measure of what the adaptive climb buys, and flat by construction on single particles."),
    ("Two-channel candidate matching",
     "A TICLCandidate should be matched on calo shared energy AND tracker shared hits at once; the payload for "
     "that is the natural next extension of the shared-hits type."),
    ("Interaction-vertex association for primary vertices",
     "A vertex should be associated to the graph Interaction vertex rather than to particle branches. The present "
     "PV numbers are mechanically correct but aimed at the wrong truth object."),
]
# Metric order drives the page order, so a reader meets efficiency before its failure modes.
METRIC_ORDER = ["composition", "efficiency", "duplicate", "splitrate", "recopurity", "fakerate",
                "nocandidate", "nolevelcandidate", "contaminated",
                "pileuprate", "resolution"]
# caloeta sits next to eta on purpose: the two answer "where was the branch root
# produced" and "where did the branch reach the calorimeter", and for anything but the
# caloBoundary level they are different questions with different answers.
VARIABLE_ORDER = ["pt", "eta", "caloeta", "phi", "nhits", "vertpos", "zpos", "dxy", "dz", "depth",
                  "root_footprint_fraction", "flavour"]
# Axes whose bins are named categories rather than numbers, drawn as grouped bars.
CATEGORICAL = ["reason"]
# Gaussian slice fits, ordered so bias is read before width for each quantity.
RESOLUTION_ORDER = [
    "ptres_vs_eta_Mean", "ptres_vs_eta_Sigma",
    "ptres_vs_pt_Mean", "ptres_vs_pt_Sigma",
    "etares_vs_eta_Mean", "etares_vs_eta_Sigma",
    "phires_vs_eta_Mean", "phires_vs_eta_Sigma",
]
_RES_RE = re.compile(r"^(?P<base>\w+res_vs_\w+)_(?P<stat>Mean|Sigma)$")
RESOLUTION_SOURCES = ["ptres_vs_eta", "ptres_vs_pt", "etares_vs_eta", "phires_vs_eta"]
# A Gaussian fitted to a slice with a handful of entries returns a width that is not a
# resolution. Below this many entries the point is dropped, not drawn.
MIN_SLICE_ENTRIES = 20
# A ratio formed from a handful of entries is noise with a large error bar, not a
# measurement. Bins whose DENOMINATOR is below this are not drawn.
MIN_DENOM_ENTRIES = 10
# Which num_* histogram is the denominator of each metric, so a bin can be dropped when
# there was nothing there to divide by.
DENOMINATOR = {
    "efficiency": "num_simul",
    "duplicate": "num_simul",
    "splitrate": "num_simul",
    "recopurity": "num_reco",
    "fakerate": "num_reco",
    "nocandidate": "num_reco",
    "nolevelcandidate": "num_reco",
    "contaminated": "num_reco",
    "pileuprate": "num_reco",
}
CATEGORICAL_MEANING = {
    "reason": (
        "the Geant4 process that CREATED the branch root, read from the VertexReason of its production vertex. "
        "GenOnly is its own bin, not Unknown: a GEN-only production vertex has no SimVertex and therefore no "
        "Geant4 process at all. In a pileup sample it dominates, because collapsePileupGen replaces each pileup "
        "interaction with one GEN-only vertex carrying all of its stable particles. "
        "Primary means the particle came straight from the hard scatter; every other value is a secondary made "
        "in the detector material. This axis exists only because the graph keeps the process that made each "
        "particle: a frozen TrackingParticle or CaloParticle does not carry it."
    ),
}
_ME_RE = re.compile(
    r"^(?P<metric>efficiency_cumulative|efficiency|recopurity|fakerate|nocandidate|nolevelcandidate"
    r"|contaminated|duplicate"
    r"|splitrate|pileuprate)"
    r"_vs_(?P<var>\w+)$"
)


def hist_arrays(h):
    """Bin edges, contents and errors of a TH1 as numpy arrays."""
    n = h.GetNbinsX()
    edges = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, n + 2)])
    values = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
    errors = np.array([h.GetBinError(i) for i in range(1, n + 1)])
    return edges, values, errors


def bin_labels(h):
    """The alphanumeric bin labels of a categorical axis, or None if it has none."""
    axis = h.GetXaxis()
    labels = [axis.GetBinLabel(i) for i in range(1, h.GetNbinsX() + 1)]
    return labels if any(labels) else None


# The two reconstructions of the same event. They are never pooled: each gets its own
# pages, so a comparison is between pages and not inside a plot.
FLAVOURS = ["Offline", "HLT"]


# Acceptance-region sub-folders, mirroring truth::kEtaRegionFolders. Same ME names as the
# inclusive folder, so they need no other special handling.
REGION_FOLDERS = {"etaLt15", "eta15to30", "eta30to45"}
REGION_MEANING = {
    "etaLt15": "|eta| below 1.5, the barrel. No trackster exists here, so a calorimetric efficiency in this "
               "region is an acceptance statement and not a reconstruction one.",
    "eta15to30": "|eta| 1.5 to 3.0, the HGCAL endcap, where the calorimetric reconstruction actually runs.",
    "eta30to45": "|eta| 3.0 to 4.5, forward. Usually a handful of objects, so read the denominator before the ratio.",
}



def _source_of(relpath, start_marker, end_marker=None, max_lines=80):
    """The actual source text of a definition, read at plot time from the release.

    A page that cites "SomeFile.cc:2819" is wrong the moment a line is inserted above it,
    and the reader cannot tell. Embedding the text means the page always shows the
    definition that produced the numbers on it, and there is nothing to keep in sync.
    """
    for base in (os.environ.get("CMSSW_BASE"), os.environ.get("CMSSW_RELEASE_BASE")):
        if not base:
            continue
        path = os.path.join(base, "src", relpath)
        if not os.path.exists(path):
            continue
        with open(path, errors="ignore") as fh:
            lines = fh.read().splitlines()
        try:
            i = next(n for n, l in enumerate(lines) if start_marker in l)
        except StopIteration:
            continue
        out = []
        for l in lines[i:i + max_lines]:
            out.append(l)
            if end_marker is not None and len(out) > 1 and end_marker in l:
                break
        return "\n".join(out), os.path.join("src", relpath)
    return None, None


def _configured_values():
    """The thresholds and species actually configured, imported rather than transcribed."""
    rows = []
    try:
        import Validation.TruthInfo.truthBranchValidation_cff as _cff
        for d in _cff._domains:
            th = ", ".join(f"{k} = {v}" for k, v in sorted(d["thresholds"].items()))
            rows.append((d["name"], th or "(none)"))
    except Exception as exc:
        rows.append(("could not import the validation configuration", str(exc)))
    seeds = []
    try:
        from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet
        ps = postProcessingPSet(name="TenTau_E_15_500_pythia8_cfi")
        seeds.append(("seedPdgIds", list(ps.seedPdgIds)))
        seeds.append(("reconstructablePdgIds", list(ps.reconstructablePdgIds)))
    except Exception as exc:
        seeds.append(("could not import the selection preset", str(exc)))
    return rows, seeds


def definitions_html():
    """A section showing the real definitions: configured values and predicate source."""
    out = ["<h2>Definitions, as the code actually has them</h2>",
           "<p>Read at plot time from the release this file was made with, so nothing here can drift "
           "out of date against the numbers on these pages. No file-and-line citations: those go stale "
           "silently the moment a line is inserted above them.</p>"]
    rows, seeds = _configured_values()
    out.append("<h3>Thresholds each domain is judged by</h3><ul class='idx'>")
    for name, th in rows:
        out.append(f"<li><span class='f'>{name}</span> {th}</li>")
    out.append("</ul>")
    out.append("<h3>Selection preset, TenTau shown as the example</h3><ul class='idx'>")
    for k, v in seeds:
        out.append(f"<li><span class='f'>{k}</span> {v}</li>")
    out.append("</ul>")
    for title, relpath, marker, end in (
            ("Level membership, the per-particle predicate",
             "PhysicsTools/TruthInfo/interface/TruthLevels.h", "inline bool atLevel(", "  }"),
            ("The antichain reduction every level goes through",
             "PhysicsTools/TruthInfo/interface/TruthLevels.h", "levelAntichain(Graph const&", "return antichain;"),
            ("The visible final state of the signal",
             "PhysicsTools/TruthInfo/interface/TruthLevels.h", "reconstructableFromSignal(Graph const&",
             "return legs;"),
            ("Acceptance regions",
             "Validation/TruthInfo/interface/TruthBranchHistoProducerAlgo.h", "inline EtaRegion etaRegionOf(",
             "  }")):
        text, shown = _source_of(relpath, marker, end)
        if text:
            import html as _html
            out.append(f"<h3>{title}</h3><p class='f'>{shown}</p>"
                       f"<pre style='background:#f6f6f6;padding:8px;overflow-x:auto'>{_html.escape(text)}</pre>")
    return "".join(out)


def region_label(category):
    """The acceptance range this plot covers, spelled out for the image itself.

    Without it a region plot and the inclusive one carry the SAME title, so a PNG on its
    own is ambiguous: the range only survives in the filename and the page it sits on.
    Inclusive says so explicitly rather than staying silent, since silence is what made
    the region plots indistinguishable in the first place.
    """
    region = category.rsplit("/", 1)[-1] if "/" in category else ""
    return {
        "etaLt15": "|eta| < 1.5",
        "eta15to30": "1.5 < |eta| < 3.0",
        "eta30to45": "3.0 < |eta| < 4.5",
    }.get(region, "all |eta|")


def discover(tfile):
    """Yield (flavour, category, folder, TDirectory) for every directory with histograms.

    The DQM path is TruthInfo/<flavour>/<category>/<collection>_<workingPoint>. A file
    written before the flavour level existed has no such component and is reported as
    Offline, so old files still plot.
    """

    def walk(directory, path):
        holds = False
        for key in directory.GetListOfKeys():
            # Classify by the key's class name: ReadObj deserializes the whole payload
            # and is only needed for directories the walk descends into.
            cls = ROOT.TClass.GetClass(key.GetClassName())
            if cls and cls.InheritsFrom("TDirectory"):
                yield from walk(key.ReadObj(), path + [key.GetName()])
            elif cls and cls.InheritsFrom("TH1"):
                holds = True
        if holds and len(path) >= 2:
            # An acceptance region is a sub-folder of the collection folder, carrying the
            # same ME names. Fold it into the CATEGORY so every downstream page, caption
            # and ratio works unchanged and the regions simply appear as their own
            # entries rather than being silently skipped by the collection_wp split.
            if path[-1] in REGION_FOLDERS and len(path) >= 3:
                flavour = path[-4] if len(path) >= 4 and path[-4] in FLAVOURS else "Offline"
                yield flavour, path[-3] + "/" + path[-1], path[-2], directory
            else:
                flavour = path[-3] if len(path) >= 3 and path[-3] in FLAVOURS else "Offline"
                yield flavour, path[-2], path[-1], directory

    yield from walk(tfile, [])


def collect(files):
    """{category: {collection: {metric: {var: {wp: (edges, values, errors)}}}}}"""
    data = {}
    for fname in files:
        tfile = ROOT.TFile.Open(fname)
        if not tfile or tfile.IsZombie():
            print(f"cannot open {fname}", file=sys.stderr)
            continue
        for flavour, category, folder, folderDir in discover(tfile):
            # Folder is "<collection>_<workingPoint>"; split on the LAST underscore so a
            # collection label containing underscores survives.
            if "_" not in folder:
                continue
            collection, wp = folder.rsplit("_", 1)
            category = f"{flavour}/{category}"
            for key in folderDir.GetListOfKeys():
                # One deserialization per key; every branch below reads this object.
                obj = key.ReadObj()
                # The per-process population is the denominator of the categorical
                # ratios; it is carried along so a bar can be dropped when the process
                # simply does not occur, rather than drawn as a zero efficiency.
                name = key.GetName()
                if name.startswith("num_simul_") or name.startswith("num_reco_"):
                    if obj.InheritsFrom("TH1") and not obj.InheritsFrom("TH2"):
                        _, counts, _ = hist_arrays(obj)
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_denom", {})
                            .setdefault(name, {})[wp]
                        ) = counts
                if key.GetName() == "num_simul_reason":
                    if obj.InheritsFrom("TH1"):
                        _, counts, _ = hist_arrays(obj)
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_counts", {})
                            .setdefault("reason", {})[wp]
                        ) = (counts, bin_labels(obj))
                    continue
                if key.GetName() in RESOLUTION_SOURCES:
                    if obj.InheritsFrom("TH2"):
                        proj = obj.ProjectionX()
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_slices", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = (
                            np.array([proj.GetBinContent(i) for i in range(1, proj.GetNbinsX() + 1)]),
                            0.5 * (obj.GetYaxis().GetXmax() - obj.GetYaxis().GetXmin()),
                            (obj.GetYaxis().GetXmax() - obj.GetYaxis().GetXmin()) / obj.GetNbinsY(),
                        )
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("_residual", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = hist_arrays(obj.ProjectionY())
                    continue
                res = _RES_RE.match(key.GetName())
                if res:
                    if obj.InheritsFrom("TH1") and obj.GetEntries() > 0:
                        (
                            data.setdefault(category, {})
                            .setdefault(collection, {})
                            .setdefault("resolution", {})
                            .setdefault(key.GetName(), {})[wp]
                        ) = hist_arrays(obj)
                    continue
                match = _ME_RE.match(key.GetName())
                if not match:
                    continue
                if not obj.InheritsFrom("TH1") or obj.GetEntries() == 0:
                    continue
                metric, var = match.group("metric"), match.group("var")
                if var in CATEGORICAL:
                    (
                        data.setdefault(category, {})
                        .setdefault(collection, {})
                        .setdefault(metric, {})
                        .setdefault(var, {})[wp]
                    ) = hist_arrays(obj) + (bin_labels(obj),)
                    continue
                (
                    data.setdefault(category, {})
                    .setdefault(collection, {})
                    .setdefault(metric, {})
                    .setdefault(var, {})[wp]
                ) = hist_arrays(obj)
        tfile.Close()
    return data


def _fit_ok(wp, values, slices, is_sigma=False, errors=None):
    """Mask of slices whose Gaussian fit can be believed.

    Three ways a slice fit is worthless: too few entries to constrain it, a width wider
    than the histogram it was fitted in, and a width narrower than one bin. The first
    two are runaway fits, the third is a fit that collapsed onto a single bin.
    """
    ok = np.ones(len(values), dtype=bool)
    if slices is None or wp not in slices:
        return ok
    counts, half_range, bin_width = slices[wp]
    if len(counts) == len(values):
        ok &= counts >= MIN_SLICE_ENTRIES
    ok &= np.abs(values) <= half_range
    if is_sigma:
        ok &= values > bin_width
    if errors is not None:
        # An error comparable to the value means the fit did not converge on anything.
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(values != 0, np.abs(errors / values), np.inf)
        ok &= rel <= 0.5
    return ok


def broken(values, keep):
    """The series with the dropped bins as NaN, keeping the x grid contiguous.

    A dropped bin must BREAK the line, not be skipped over: the segment matplotlib would
    otherwise draw across it reads as a measurement in a region that has none, such as the
    barrel gap where HGCAL has no acceptance. NaN lifts the pen and draws no marker.
    """
    out = np.asarray(values, dtype=float).copy()
    out[~np.asarray(keep, dtype=bool)] = np.nan
    return out


def plot_metric(category, collection, metric, var, per_wp, outdir, index, slices=None, denom=None,
                order=None, reference=None, paired=None, note=None):
    """One overlay plot with a ratio panel against the reference series.

    The series are working points for the reco-driven metrics and graph levels for the
    truth-driven ones; order and reference name which comparison this plot makes.
    paired holds a second curve per series (the cumulative efficiency), drawn in the
    series colour with open markers and a dashed line; the ratio panel stays on the
    primary curves. note is the one-line match criterion drawn under the legend.
    """
    order = order or WP_ORDER
    wps = [w for w in order if w in per_wp] + [w for w in sorted(per_wp) if w not in order]
    if not wps:
        return None
    reference = reference or REFERENCE_WP
    if reference not in per_wp:
        reference = wps[0]

    is_sigma = var.endswith("_Sigma")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Working points can lie on top of one another (the adaptive points differ by ~0.002
    # here), so vary marker AND linestyle: colour alone hides a curve completely.
    markers = SERIES_MARKERS
    styles = SERIES_STYLES
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.07)
    )
    fig.subplots_adjust(top=0.88, bottom=0.16)

    means = {}
    for i, wp in enumerate(wps):
        edges, values, errors = per_wp[wp]
        centers = 0.5 * (edges[:-1] + edges[1:])
        # A measured zero is data: an efficiency that falls to zero inside acceptance, a
        # fake rate of zero on a clean sample, a fitted mean that sits at zero. Only a bin
        # with no entries at all is empty, and such a bin carries both a zero value and a
        # zero uncertainty. The denominator cut below removes the thinly populated bins.
        filled = np.isfinite(values) & ((values != 0) | (errors > 0))
        filled = filled & _fit_ok(wp, values, slices, is_sigma, errors if metric == "resolution" else None)
        if denom is not None and wp in denom and len(denom[wp]) == len(values):
            filled = filled & (denom[wp] >= MIN_DENOM_ENTRIES)
        means[wp] = float(values[filled].mean()) if filled.any() else 0.0
        ax.errorbar(
            centers,
            broken(values, filled),
            yerr=broken(errors, filled),
            fmt=markers[i % len(markers)],
            linestyle=styles[i % len(styles)] if paired is None else "-",
            markersize=marker_size(markers[i % len(markers)]),
            markerfacecolor="none" if (i and paired is None) else None,
            linewidth=1.4,
            alpha=0.85,
            color=colors[i % len(colors)],
            label=wp,
        )
        if paired is not None and wp in paired:
            # The cumulative partner of this series: same colour and shape so the pair
            # reads as one level, open marker and dashed line so the two are distinct.
            p_edges, p_values, p_errors = paired[wp]
            p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
            p_filled = np.isfinite(p_values) & ((p_values != 0) | (p_errors > 0))
            if denom is not None and wp in denom and len(denom[wp]) == len(p_values):
                p_filled = p_filled & (denom[wp] >= MIN_DENOM_ENTRIES)
            ax.errorbar(
                p_centers,
                broken(p_values, p_filled),
                yerr=broken(p_errors, p_filled),
                fmt=markers[i % len(markers)],
                linestyle="--",
                markersize=marker_size(markers[i % len(markers)]),
                markerfacecolor="none",
                linewidth=1.2,
                alpha=0.7,
                color=colors[i % len(colors)],
                label=f"{wp} cumulative",
            )

    label, meaning, formula = METRICS[metric]
    if metric == "resolution":
        # Residuals are not bounded to [0, 1]; a fixed range would push every point off
        # the axis. Scale to the data, keeping zero visible so a bias is readable.
        def _shown(w):
            v = per_wp[w][1]
            return v[(v != 0) & _fit_ok(w, v, slices, is_sigma)]

        shown = [s for s in (_shown(w) for w in wps) if s.size]
        allv = np.concatenate(shown or [np.zeros(1)])
        span = float(np.abs(allv).max()) if allv.size else 1.0
        ax.set_ylim(min(0.0, float(allv.min()) * 1.3 if allv.size else 0.0), span * 1.35 if span else 1.0)
        # The momentum residual is relative and dimensionless, the angular ones are
        # differences, so only the azimuth carries a unit.
        label = ("Mean" if var.endswith("_Mean") else "Sigma") + RESIDUAL_UNIT.get(var.split("res_vs_", 1)[0], "")
    # The plot title stays generic. A bin-averaged summary in the title reads as a
    # conclusion the plot has not earned, so the measured numbers go in the README
    # caption instead, where they can be qualified.
    title = f"{label} vs {var}" if metric != "resolution" else var.replace("_", " ")
    ref = means.get(reference)
    others = [means[w] for w in wps if w != reference]
    if ref and others:
        rest = sum(others) / len(others)
        delta = (rest - ref) / ref * 100.0 if ref else 0.0
        caption = (f"{title}, {region_label(category)}. Bin-averaged over filled bins: others {rest:.2f}, "
                   f"{reference} {ref:.2f} ({delta:+.0f}%).")
    else:
        caption = f"{title}, {region_label(category)}"

    fig.suptitle(title, fontsize=16, y=0.965)
    # Centred on the MAIN pad: the CMS top location hangs the title from the top of the
    # axes, so a long one runs down past the pad and into the ratio panel.
    ax.set_ylabel(label, fontsize=AXIS_TITLE_SIZE, loc="center")
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    if metric != "resolution":
        ax.set_ylim(0.0, 1.15)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    # Paired pages carry twice the entries, so the legend wraps instead of overflowing
    # into the x label at the figure's right edge.
    # Three columns once the level pairs push past eight entries: a fourth column runs
    # into the x label at the figure's right edge.
    ncol = min(len(labels), 4 if len(labels) <= 8 else 3)
    fig.legend(handles, labels, fontsize=13 if len(labels) <= 4 else 11, loc="lower center",
               ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.02))
    # The legend grows a row at a time and the x title, which carries the unit, is written
    # in the same band; the pads move up so the two do not overlap.
    fig.subplots_adjust(bottom=0.16 + 0.025 * max(0, -(-len(labels) // ncol) - 1))
    if note:
        # The match criterion the numerator was counted with, so the plot carries its
        # own definition; the sources of the thresholds are in DEFINITIONS.md.
        fig.text(0.5, 0.002, note, ha="center", va="bottom", fontsize=9, color="0.35")
    ax.tick_params(labelbottom=False)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel=f"Phase-2 Simulation, {region_label(category)}", fontsize=15)

    # Ratio panel: only where the reference has a value, so an empty reference bin does
    # not manufacture a spike.
    ratio_values = []
    if reference in per_wp:
        ref_edges, ref_values, _ = per_wp[reference]
        ref_centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
        for i, wp in enumerate(wps):
            if wp == reference:
                continue
            _, values, _ = per_wp[wp]
            # The reference has to be non-zero to divide by it. The series itself may be
            # zero: a ratio of zero is a result, not a missing point.
            ok = np.isfinite(ref_values) & np.isfinite(values) & (ref_values != 0)
            ok = ok & _fit_ok(reference, ref_values, slices, is_sigma) & _fit_ok(wp, values, slices, is_sigma)
            if denom is not None:
                for w in (reference, wp):
                    if w in denom and len(denom[w]) == len(values):
                        ok = ok & (denom[w] >= MIN_DENOM_ENTRIES)
            if ok.any():
                ratio = np.full(len(values), np.nan)
                ratio[ok] = values[ok] / ref_values[ok]
                ratio_values.extend(ratio[ok].tolist())
                # Same marker, colour and line style as the main pad, so a series is
                # recognised in the ratio without going back to the legend.
                rax.plot(
                    ref_centers,
                    ratio,
                    marker=markers[i % len(markers)],
                    linestyle=styles[i % len(styles)] if paired is None else "-",
                    markersize=marker_size(markers[i % len(markers)]),
                    markerfacecolor="none" if (i and paired is None) else None,
                    linewidth=1.4,
                    alpha=0.85,
                    color=colors[i % len(colors)],
                )
    # Scale to the data when a ratio leaves the default window, rather than drawing an
    # empty panel that reads as "no points" instead of "points off scale".
    rax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2)
    rax.set_ylabel(f"ratio to {reference}", fontsize=RATIO_TITLE_SIZE)
    rax.set_ylim(0.0, max(2.0, max(ratio_values) * 1.15 if ratio_values else 2.0))
    rax.tick_params(labelsize=TICK_LABEL_SIZE)
    xvar = var.rsplit("_vs_", 1)[-1].split("_")[0] if "_vs_" in var else var
    # The two pads share the x axis, so the title is written once, under the ratio.
    rax.set_xlabel(axis_title(xvar), fontsize=AXIS_TITLE_SIZE)
    # Axes booked with symlog bin edges must be DRAWN symlog, or the log ladder is
    # squashed into the right-hand sliver of a linear axis and reads as an empty plot.
    # Linear below the threshold so the entries at exactly 0 stay visible: on DY 20.5% of
    # the signal level sits at pt exactly 0.
    if xvar in SYMLOG_AXES:
        _lin, _max = SYMLOG_AXES[xvar]
        for _a in (rax, ax):
            _a.set_xscale("symlog", linthresh=_lin)
            # Clamp at 0: symlog would otherwise mirror the decades into negative x.
            _a.set_xlim(0.0, _max)
    if xvar == "flavour":
        rax.set_xticks([i + 0.5 for i in range(len(FLAVOUR_BINS))])
        rax.set_xticklabels(FLAVOUR_BINS)
    if xvar in AXIS_RANGE:
        rax.set_xlim(*AXIS_RANGE[xvar])
    rax.grid(alpha=0.3)

    name = f"{index:02d}_{category.split('/')[-1]}_{collection}_{metric}_vs_{var}.png"
    fig.savefig(os.path.join(outdir, category.split("/", 1)[0], metric, name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_categorical(category, collection, metric, var, per_wp, counts, outdir, index,
                     order=None, reference=None):
    """Grouped horizontal bars, one group per named category, one bar per series.

    Categories the sample does not populate are dropped rather than drawn at zero: a
    process that never happened is not an inefficiency.
    """
    order = order or WP_ORDER
    wps = [w for w in order if w in per_wp] + [w for w in sorted(per_wp) if w not in order]
    if not wps:
        return None
    reference = reference or REFERENCE_WP
    labels = next((per_wp[w][3] for w in wps if per_wp[w][3]), None)
    if labels is None:
        return None

    population = None
    if counts:
        population = counts.get(reference, next(iter(counts.values())))[0]
    keep = [i for i in range(len(labels))
            if (population is None and any(per_wp[w][1][i] > 0 for w in wps)) or
               (population is not None and population[i] > 0)]
    if not keep:
        return None
    # Most populated process first, so the rows the reader should trust come first and
    # a one-entry category cannot sit between two well-populated ones.
    if population is not None:
        keep.sort(key=lambda k: population[k], reverse=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(keep) + 3.6))
    # Margins fixed in inches, not in figure fractions: the figure height grows with the
    # number of categories, so a fractional bottom margin would shrink the label space.
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.9 / height, bottom=1.6 / height)

    y = np.arange(len(keep))
    barh = 0.8 / len(wps)
    for i, wp in enumerate(wps):
        _, values, errors, _ = per_wp[wp]
        offset = (i - (len(wps) - 1) / 2.0) * barh
        ax.barh(y + offset, [values[k] for k in keep], height=barh * 0.92,
                xerr=[errors[k] for k in keep], color=colors[i % len(colors)],
                edgecolor="none", alpha=0.9, label=wp, error_kw=dict(lw=1, ecolor="0.3"))

    label, meaning, formula = METRICS[metric]
    title = f"{label} vs {var}"
    # The category population is what makes a bar readable, so it is written next to
    # the label instead of being left to the reader to guess from the error bar.
    ticks = []
    for k in keep:
        if population is not None:
            ticks.append(f"{labels[k]}  (N={int(population[k])})")
        else:
            ticks.append(labels[k])
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(label, fontsize=AXIS_TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=16, y=0.965)
    # Below everything: a legend inside the axes covers the least populated rows, which
    # are still real measurements, so it goes under the x label in the reserved margin.
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=13, loc="lower center", ncol=len(lbls), frameon=False,
               bbox_to_anchor=(0.5, 0.15 / height))
    hep.cms.label(ax=ax, llabel="Private Work", rlabel=f"Phase-2 Simulation, {region_label(category)}", fontsize=15)

    top = max(keep, key=lambda k: population[k]) if population is not None else keep[0]
    caption = (f"{title}. Most populated process: {labels[top]} "
               f"(N={int(population[top])}) at {per_wp[wps[0]][1][top]:.2f} for {wps[0]}."
               if population is not None else title)

    name = f"{index:02d}_{category.split('/')[-1]}_{collection}_{metric}_vs_{var}.png"
    fig.savefig(os.path.join(outdir, category.split("/", 1)[0], metric, name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_residual(category, collection, source, per_wp, outdir, index):
    """The residual distribution itself, overlaid across working points.

    The Gaussian slice fit summarises this distribution; when the distribution is not
    Gaussian the fit says nothing and only the distribution does.
    """
    wps = [w for w in WP_ORDER if w in per_wp] + [w for w in sorted(per_wp) if w not in WP_ORDER]
    if not wps:
        return None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = SERIES_STYLES
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(top=0.88, bottom=0.16)

    cores = {}
    for i, wp in enumerate(wps):
        edges, values, _ = per_wp[wp]
        total = values.sum()
        if total <= 0:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Fraction inside +-10%, a scale-free statement about how peaked it is that does
        # not depend on a fit converging.
        cores[wp] = float(values[np.abs(centers) <= 0.1].sum() / total)
        hep.histplot(broken(values / total, values > 0), edges, ax=ax, label=wp, yerr=False,
                     color=colors[i % len(colors)], linestyle=styles[i % len(styles)], linewidth=1.6)

    ax.set_yscale("log")
    ax.set_xlabel(RESIDUAL_TITLE.get(source.split("res_vs_", 1)[0], "reco - truth"), fontsize=AXIS_TITLE_SIZE)
    ax.set_ylabel("fraction of matched pairs per bin", fontsize=AXIS_TITLE_SIZE, loc="center")
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, frameon=False)
    fig.suptitle(f"{source} residual distribution", fontsize=16, y=0.965)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel=f"Phase-2 Simulation, {region_label(category)}", fontsize=15)

    ref = cores.get(REFERENCE_WP)
    caption = (f"{source} residual distribution, area normalised. Fraction within 10%: "
               + ", ".join(f"{w} {cores[w]:.2f}" for w in wps if w in cores) + ".") if cores else source
    name = f"{index:02d}_{category.split('/')[-1]}_{collection}_{source}_distribution.png"
    fig.savefig(os.path.join(outdir, category.split("/", 1)[0], "resolution", name), dpi=150)
    plt.close(fig)
    return name, caption


def plot_composition(category, collection, counts, outdir, index, reference=None):
    """What the selected truth branches ARE, by the process that created them."""
    entry = counts.get(reference or REFERENCE_LEVEL) or next(iter(counts.values()))
    values, labels = entry
    if labels is None or values.sum() <= 0:
        return None
    keep = [i for i in range(len(labels)) if values[i] > 0]
    order = sorted(keep, key=lambda k: values[k], reverse=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(order) + 3.4))
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.9 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight())
    frac = np.array([values[k] for k in order]) / values.sum()
    ax.barh(np.arange(len(order)), frac, color=colors[0], alpha=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{labels[k]}  (N={int(values[k])})" for k in order], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of selected truth branches", fontsize=AXIS_TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Selected truth branches by creation process", fontsize=16, y=0.965)
    hep.cms.label(ax=ax, llabel="Private Work", rlabel=f"Phase-2 Simulation, {region_label(category)}", fontsize=15)

    caption = ("Composition of the truth-branch denominator by the Geant4 process that created each branch root. "
               f"Leading process {labels[order[0]]} at {frac[0]*100:.0f}% of {int(values.sum())} branches.")
    name = f"{index:02d}_{category.split('/')[-1]}_{collection}_composition_by_reason.png"
    fig.savefig(os.path.join(outdir, category.split("/", 1)[0], "composition", name), dpi=150)
    plt.close(fig)
    return name, caption


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outputDir", default="plots")
    ap.add_argument("--sample", default="ttbar, no pileup, D122")
    ap.add_argument("--title", default="MC-truth graph validation")
    ap.add_argument("--jobs", default=0, type=int,
                    help="Processes for drawing plots. 0 (default) uses every core; 1 draws in the main "
                         "process, which is what you want when reading a traceback.")
    ap.add_argument("--gallery", default="truth-validation",
                    help="orbit gallery name; also the .orbit target, so two samples can be published side by side")
    args = ap.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)
    # One real directory per metric, so the gallery browses as folders and not as one
    # flat list of sixty files.
    for flavour in FLAVOURS:
        for metric in METRIC_ORDER:
            os.makedirs(os.path.join(args.outputDir, flavour, metric), exist_ok=True)
    data = collect(args.files)
    if not data:
        print("no populated monitor elements found", file=sys.stderr)
        return 1

    written = []
    metric_tasks = []
    index = 1
    for category in sorted(data):
        for collection in sorted(data[category]):
            all_counts = data[category][collection].get("_counts", {})
            all_slices = data[category][collection].get("_slices", {})
            all_denom = data[category][collection].get("_denom", {})
            all_residual = data[category][collection].get("_residual", {})
            if "reason" in all_counts:
                result = plot_composition(category, collection, all_counts["reason"], args.outputDir, index)
                if result:
                    name, caption = result
                    written.append({"_ord": index, "category": category, "collection": collection, "metric": "composition",
                                    "var": "reason", "png": name, "caption": caption})
                    index += 1
            for metric in METRIC_ORDER:
                per_metric = data[category][collection].get(metric, {})
                # Truth-driven metrics compare graph levels; everything else compares
                # working points.
                if metric in TRUTH_METRICS:
                    series_order, series_ref = TRUTH_SUFFIXES, REFERENCE_LEVEL
                else:
                    series_order, series_ref = WP_ORDER, REFERENCE_WP
                # The cumulative numerator pairs with the individual one on the
                # efficiency page rather than getting a page of its own.
                cumulative = (data[category][collection].get("efficiency_cumulative", {})
                              if metric == "efficiency" else {})
                for var in VARIABLE_ORDER:
                    if var not in per_metric:
                        continue
                    _criterion = (MATCH_CRITERIA.get(category.split("/")[-1])
                                  if metric in TRUTH_METRICS else None)
                    # Queued rather than drawn here: every argument is numpy arrays and
                    # plain values, since hist_arrays converts the TH1 before this point,
                    # so the work pickles and a worker process can do it. Drawing is the
                    # bulk of the runtime and is what parallelises.
                    metric_tasks.append((
                        (category, collection, metric, var, per_metric[var], args.outputDir, index,
                         None,
                         all_denom.get(f"{DENOMINATOR.get(metric, '')}_{var}"),
                         series_order, series_ref, cumulative.get(var),
                         _criterion[0] if _criterion else None),
                        {"_ord": index, "category": category, "collection": collection, "metric": metric, "var": var},
                    ))
                    # Unconditional now: the index only has to be unique per plot, and the
                    # worker decides whether the plot is drawable.
                    index += 1
                if metric == "resolution":
                    for source in RESOLUTION_SOURCES:
                        if source not in all_residual:
                            continue
                        result = plot_residual(
                            category, collection, source, all_residual[source], args.outputDir, index
                        )
                        if result:
                            name, caption = result
                            written.append({"_ord": index, "category": category, "collection": collection, "metric": metric,
                                            "var": source, "png": name, "caption": caption})
                            index += 1
                    for var in RESOLUTION_ORDER:
                        if var not in per_metric:
                            continue
                        base = var.rsplit("_", 1)[0]
                        # Queued like every other plot_metric call, so the resolution
                        # pages draw in the process pool too.
                        metric_tasks.append((
                            (category, collection, metric, var, per_metric[var], args.outputDir, index,
                             all_slices.get(base), None, None, None, None, None),
                            {"_ord": index, "category": category, "collection": collection, "metric": metric,
                             "var": var},
                        ))
                        index += 1
                    continue
                for var in CATEGORICAL:
                    if var not in per_metric:
                        continue
                    result = plot_categorical(
                        category, collection, metric, var, per_metric[var],
                        all_counts.get(var), args.outputDir, index,
                        order=series_order, reference=series_ref,
                    )
                    if result:
                        name, caption = result
                        written.append({"_ord": index, "category": category, "collection": collection, "metric": metric,
                                        "var": var, "png": name, "caption": caption})
                        index += 1

    # Draw the queued plots in a process pool. One task is one plot, which is the same
    # granularity SimpleValidation uses for the standard validation, and the natural unit
    # here since each writes its own PNG and shares nothing.
    if metric_tasks:
        nproc = args.jobs if args.jobs > 0 else os.cpu_count()
        nproc = max(1, min(nproc, len(metric_tasks)))
        if nproc == 1:
            results = [plot_metric(*t[0]) for t in metric_tasks]
        else:
            import multiprocessing
            with multiprocessing.Pool(nproc) as pool:
                results = pool.starmap(plot_metric, [t[0] for t in metric_tasks], chunksize=4)
        for (_, meta), result in zip(metric_tasks, results):
            if result:
                name, caption = result
                written.append(dict(meta, png=name, caption=caption))

    # Restore creation order: the queued plots are appended after the loop, so without
    # this the categorical plots drawn inline would jump ahead of them on the page.
    written.sort(key=lambda e: e["_ord"])
    by_page = {}
    for entry in written:
        flavour = entry["category"].split("/", 1)[0]
        by_page.setdefault((flavour, entry["metric"]), []).append(entry)

    # What the curves of one plot are, per metric family. The truth-driven metrics
    # overlay graph levels, the reco-driven ones working points.
    def overlay_note(metric):
        if metric in TRUTH_METRICS:
            return ("Each plot overlays the branch LEVELS of the truth graph, the a priori definitions of what "
                    "one truth object is (stableLegsFromUpstream, caloBoundary, stableDecayProducts, "
                    "hardProcess, reconstructableFromSignal), plus three more series: signal, whose denominator "
                    "is the preset SEED objects "
                    "among the selected roots, so with a selection preset it is the signal object's own "
                    "efficiency (the tau, not its decay legs), signalNoSelection, the same seed objects with "
                    "no branch-selector cut at all, so the efficiency is quoted against every seed in the "
                    "event and the gap to signal is what the selection removed. Every one of these is an "
                    "ANTICHAIN, so no object is counted twice: a denominator holding both a particle and "
                    "its own daughter would count the same energy twice. A "
                    "composite domain has a single folder named by its vertex resolution. On the efficiency "
                    "page each series is a pair: individual (filled, solid) means a single reco object covers "
                    "the truth object, cumulative (open, dashed, same colour) means all reco objects of the "
                    "collection together do; a multi-prong decay separates the two. The "
                    f"lower panel is the ratio to {REFERENCE_LEVEL}.")
        return ("Each plot overlays the four branch-association working points. Fixed keeps every matching "
                "branch; the Adaptive points keep the single graph level that best matches the reco object, "
                "differing only in how much branch spread they tolerate. The lower panel is the ratio to "
                f"{REFERENCE_WP}.")

    # One page per metric, each opening with the definition of the quantity it shows.
    for (flavour, metric), entries in by_page.items():
        label, meaning, formula = METRICS[metric]
        with open(os.path.join(args.outputDir, f"{flavour}_{metric}.html"), "w") as page:
            page.write(f"<!doctype html><meta charset='utf-8'><title>{label}</title><style>{STYLE}</style>")
            page.write(f"<h1>{flavour}: {label}</h1><p><a href='index.html'>back to index</a></p>")
            page.write(f"<div class='def'><b>Definition.</b> {meaning}<br><br>"
                       f"<span class='f'>{formula}</span><br><br>"
                       f"{overlay_note(metric)}</div>")
            page.write(f"<p>Sample: {args.sample}.</p>")
            used = [v for v in VARIABLE_ORDER + CATEGORICAL
                    if any(e["var"] == v or e["var"].endswith("_vs_" + v) for e in entries)]
            if used:
                page.write("<div class='def'><b>What is on the x axis.</b><ul>")
                for v in used:
                    meaning = VARIABLE_MEANING.get(v) or CATEGORICAL_MEANING.get(v, v)
                    page.write(f"<li><span class='f'>{v}</span> {meaning}</li>")
                page.write("</ul></div>")
            for category in sorted({e["category"].split("/")[-1] for e in entries}):
                if category in CATEGORY_NOTE:
                    page.write(f"<div class='def'><b>{category}.</b> {CATEGORY_NOTE[category]}</div>")
                if metric in TRUTH_METRICS and category in MATCH_CRITERIA:
                    page.write(f"<div class='def'><b>{category} match criterion.</b> "
                               f"{MATCH_CRITERIA[category][1]}</div>")
            for collection in sorted({e["collection"] for e in entries}):
                page.write(f"<h2>{collection}</h2><div class='grid'>")
                for e in [x for x in entries if x["collection"] == collection]:
                    href = f"{flavour}/{metric}/{e['png']}"
                    page.write(f"<a href='{href}'><img src='{href}' width='400'></a>")
                page.write("</div>")

    # Each folder carries its own definitions, so a reader who lands in the folder
    # without going through the index still knows what the plots in it mean.
    for (flavour, metric), entries in by_page.items():
        label, meaning, formula = METRICS[metric]
        with open(os.path.join(args.outputDir, flavour, metric, "DEFINITIONS.md"), "w") as defs:
            defs.write(f"# {label}\n\nSample: {args.sample}.\n\n## Definition\n\n{meaning}\n\n")
            defs.write(f"    {formula}\n\n## What the curves are\n\n")
            defs.write(overlay_note(metric) + "\n\n## Variables on the x axis\n\n")
            used = [v for v in VARIABLE_ORDER + CATEGORICAL
                    if any(e["var"] == v or e["var"].endswith("_vs_" + v) for e in entries)]
            for v in used:
                defs.write(f"- `{v}`: {VARIABLE_MEANING.get(v) or CATEGORICAL_MEANING.get(v, v)}\n")
            if metric in TRUTH_METRICS:
                _crit = sorted({c for c in (e["category"].split("/")[-1] for e in entries)
                                if c in MATCH_CRITERIA})
                if _crit:
                    defs.write("## Individual-match criterion, from the standard validations\n\n")
                    for _c in _crit:
                        defs.write(f"- **{_c}.** {MATCH_CRITERIA[_c][1]}\n")
                    defs.write("\n")
            defs.write("\n## Quality cuts\n\n"
                       f"- A ratio bin is drawn only if its denominator has at least {MIN_DENOM_ENTRIES} entries.\n"
                       "- A bin that is empty or suppressed breaks the line; no segment is drawn across it.\n"
                       f"- A Gaussian slice fit is drawn only if its slice has at least {MIN_SLICE_ENTRIES} entries\n"
                       "  and the fitted width is inside the fit range and wider than one bin.\n"
                       "- A named category is drawn only if the sample populated it.\n")
            _cats = sorted({c for c in (e["category"].split("/")[-1] for e in entries) if c in CATEGORY_NOTE})
            if _cats:
                defs.write("\n## Caveats by domain\n\n")
                for _c in _cats:
                    defs.write(f"- **{_c}.** {CATEGORY_NOTE[_c]}\n")
            defs.write("\n## Plots in this folder\n\n")
            for e in entries:
                defs.write(f"- `{e['png']}`: {e['caption']}\n")

    with open(os.path.join(args.outputDir, "index.html"), "w") as idx:
        idx.write(f"<!doctype html><meta charset='utf-8'><title>{args.title}</title><style>{STYLE}</style>")
        idx.write(f"<h1>{args.title}</h1><p>Sample: {args.sample}.</p>")
        idx.write("<div class='def'>Truth objects here are <b>branches</b> of the MC-truth graph, not frozen "
                  "TrackingParticles or CaloParticles. A branch is recomputed on demand from the graph, so the "
                  "same validation can be re-run against a different definition of what counts as one truth "
                  "object. Only branches passing the branch selector enter the denominators.</div>")
        for flavour in FLAVOURS:
            pages = [m for m in METRIC_ORDER if (flavour, m) in by_page]
            if not pages:
                continue
            idx.write(f"<h2>{flavour} reconstruction</h2>")
            collections = sorted({e["collection"] for (f, _), es in by_page.items() if f == flavour for e in es})
            idx.write(f"<p>Collections: {', '.join('<span class=\'f\'>' + c + '</span>' for c in collections)}</p>")
            idx.write("<ul class='idx'>")
            for metric in pages:
                label, meaning, formula = METRICS[metric]
                idx.write(f"<li><a href='{flavour}_{metric}.html'><b>{label}</b></a> "
                          f"<span class='f'>{formula}</span><br>{meaning}</li>")
            idx.write("</ul>")
        idx.write("<h2>Reading the two together</h2><p>Offline and HLT are two different reconstructions of "
                  "the SAME event, validated against the same truth objects with the same working points, so a "
                  "difference between the two pages is a difference between the reconstructions and not between "
                  "the truth definitions. They are never pooled into one plot.</p>")
        idx.write("<ul class='idx'>")
        idx.write("</ul><h2>Variables</h2><ul class='idx'>")
        for var in VARIABLE_ORDER:
            idx.write(f"<li><span class='f'>{var}</span> {VARIABLE_MEANING[var]}</li>")
        for var in CATEGORICAL:
            idx.write(f"<li><span class='f'>{var}</span> {CATEGORICAL_MEANING[var]}</li>")
        idx.write(definitions_html())
        idx.write("<h2>Truth levels: what each series is the efficiency OF</h2>"
                  "<p>The truth-driven pages overlay these. They are different DENOMINATORS over the same "
                  "event, not different measurements of one quantity, so a series sitting lower than another "
                  "usually means it is a wider denominator, not a worse reconstruction.</p><ul class='idx'>")
        for level in LEVEL_ORDER:
            if level in LEVEL_MEANING:
                idx.write(f"<li><span class='f'>{level}</span> {LEVEL_MEANING[level]}</li>")
        idx.write("</ul><h2>Quality cuts applied to every plot</h2><ul class='idx'>"
                  f"<li>A ratio bin is drawn only if its denominator has at least {MIN_DENOM_ENTRIES} entries. "
                  "A ratio built from a handful of entries is noise with a large error bar, not a measurement.</li>"
                  "<li>An empty or suppressed bin breaks the line rather than being skipped over, so a series "
                  "never draws a segment across a region it did not measure.</li>"
                  f"<li>A Gaussian slice fit is drawn only if its slice has at least {MIN_SLICE_ENTRIES} entries, "
                  "and only if the fitted width is inside the fit range and wider than one bin. The other two cases "
                  "are a fit that ran away and a fit that collapsed onto a single bin.</li>"
                  "<li>A named category is drawn only if the sample populated it. A process that never happened is "
                  "not an inefficiency.</li></ul>"
                  "<h2>Proposed plots, not yet implemented</h2>"
                  "<p>What the graph makes possible that a frozen truth object cannot answer.</p><ul class='idx'>")
        for title, why in PROPOSED:
            idx.write(f"<li><b>{title}.</b> {why}</li>")
        idx.write("</ul>")

    with open(os.path.join(args.outputDir, "README.md"), "w") as readme:
        readme.write(f"# {args.title}\n\nSample: {args.sample}.\n")
        for flavour in FLAVOURS:
            if not any(f == flavour for f, _ in by_page):
                continue
            readme.write(f"\n# {flavour} reconstruction\n")
            for metric in METRIC_ORDER:
                entries = by_page.get((flavour, metric), [])
                if not entries:
                    continue
                label, meaning, formula = METRICS[metric]
                readme.write(f"\n## {label}\n\n{meaning}\n\n    {formula}\n\n")
                readme.write(f"Folder `{flavour}/{metric}/`, definitions in "
                             f"`{flavour}/{metric}/DEFINITIONS.md`.\n\n")
                for e in entries:
                    readme.write(f"- `{flavour}/{metric}/{e['png']}`: {e['caption']}\n")
        readme.write("\n## Proposed plots, not yet implemented\n\n")
        for title, why in PROPOSED:
            readme.write(f"- **{title}.** {why}\n")

    with open(os.path.join(args.outputDir, args.gallery + ".orbit"), "w") as orbit:
        json.dump({"target": args.gallery, "title": args.title,
                   "description": f"Truth-branch association metrics ({args.sample})",
                   "icon": "chart", "access": "public"}, orbit, indent=2)

    print(f"wrote {len(written)} plots in {len(by_page)} pages to {args.outputDir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
