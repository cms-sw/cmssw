# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The single place that says WHICH reco collections are associated to truth branches
# and WITH WHICH working points. The associator producers, the DQM validators, the
# harvesters and the plotting script all import from here, so a collection is added
# in one edit and cannot drift between the four.
#
# Same shape as RecoHGCal/TICL/python/iterativeTICL_cff.py: the labels live in a
# cms.PSet rather than a plain list so an era or a process modifier can retarget a
# domain with toModify, and the instance-label lists are plain Python built by
# looping over that PSet.

import sys

import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabelsPSet

# EDProducer types whose produces<> declares a vector<ticl::Trackster>, so a trackster
# collection is recognised BY TYPE and a new TICL iteration or HLT trackster module
# joins the validation without an edit here. Verified in RecoHGCal/TICL/plugins:
# TrackstersProducer.cc:146, TracksterLinksProducer.cc:108, MergedTrackstersProducer.cc:34,
# TICLCandidateProducer.cc:208 (the post-linking trackster collection, emitted next to
# its vector<TICLCandidate>; EDM resolves the two by type from the bare module label).
# SimTrackstersProducer is deliberately absent: its tracksters are truth, not reco.
tracksterProducerTypes = (
    "TrackstersProducer",
    "TracksterLinksProducer",
    "MergedTrackstersProducer",
    "TICLCandidateProducer",
)

# Reco collections per domain. Each entry is a module label; a collection that also
# needs an instance label is written "label:instance" and split by the producers.
truthGraphRecoLabelsPSet = cms.PSet(
    tracks=cms.vstring("generalTracks"),
    vertices=cms.vstring("offlinePrimaryVertices"),
    secondaryVertices=cms.vstring("inclusiveSecondaryVertices"),
    # Particle-flow clusters, one domain per subdetector because the shared-energy
    # denominator of an associator module is one detector mask, and the efficiency gate
    # is the branch energy fraction in THAT detector. These are the calorimeter
    # constituents of the barrel PF blocks; preshower clusters are left out because the
    # truth hit index carries no preshower sim hits, so they would match nothing.
    pfClustersEcal=cms.vstring("particleFlowClusterECAL"),
    pfClustersHcal=cms.vstring("particleFlowClusterHCAL"),
    # Fallback for a job that never sees the producers: the TICL label registry. A job
    # that does run them replaces this with what its process actually schedules, via
    # setTracksterLabelsFromProcess below.
    tracksters=cms.vstring(*sorted(ticlIterLabelsPSet.labels)),
)

# The same domains reconstructed by the HLT menu. Kept as a separate PSet rather than
# extra entries in the offline one because the two are different reconstructions of the
# same event and must be compared, not pooled: they get their own producers, their own
# DQM folders and their own pages. A domain the menu does not reconstruct is left empty
# and simply produces nothing.
truthGraphHltRecoLabelsPSet = cms.PSet(
    tracks=cms.vstring("hltGeneralTracks"),
    vertices=cms.vstring("hltOfflinePrimaryVertices"),
    secondaryVertices=cms.vstring(),
    pfClustersEcal=cms.vstring(),
    pfClustersHcal=cms.vstring(),
    # Same fallback role as the offline entry, from the menu's own registry in
    # Validation/HGCalValidation. A RECO or DQM job reads the HLT tracksters from its
    # input file and so has no HLT producer to discover.
    tracksters=cms.vstring(*sorted(hltTiclIterLabelsPSet.labels)),
)

# Working points of the branch association. Fixed is the plain per-root match; the
# adaptive points differ only in how much branch spread they tolerate before
# rejecting a level.
#
# The reverse score is a fraction of the branch's own energy, so it lies in [0, 1] and a
# ceiling of 1 or above never rejects anything. AdaptiveNominal is therefore the
# unconstrained weighted argmin, and any point above it repeats it: measured on 200 no-PU
# TenTau events, a ceiling of 1.5 gave a map byte-identical to AdaptiveNominal.
truthBranchWorkingPointsPSet = cms.PSet(
    names=cms.vstring("Fixed", "AdaptiveTight", "AdaptiveNominal"),
    adaptiveReverseWeight=cms.vfloat(0.0, 1.0, 1.0),
    adaptiveMaxReverseScore=cms.vfloat(0.0, 0.6, 1.0),
)


def recoLabels(domain, flavour="offline"):
    """The reco collection labels configured for one domain and one reconstruction.

    flavour is "offline" or "hlt"; they are separate reconstructions of the same event,
    so they are never pooled.
    """
    pset = truthGraphRecoLabelsPSet if flavour == "offline" else truthGraphHltRecoLabelsPSet
    return list(getattr(pset, domain))


def _scheduledModuleNames(process):
    """The modules `process` will actually run. Only these count as discovered: loading
    a cff attaches every module it defines, so process.producers_() also holds the TICL
    iterations and the alternative superclustering that the current default leaves
    unscheduled."""
    if process.schedule is not None:
        return process.schedule.moduleNames()
    names = set()
    for container in list(process.paths.values()) + list(process.endpaths.values()):
        names |= container.moduleNames()
    return names


def tracksterLabelsFromProcess(process, flavour="offline"):
    """Every scheduled module of `process` that emits a vector<ticl::Trackster>, for one
    reconstruction, sorted so the DQM folder numbering is stable between runs.

    The two reconstructions are told apart by the hlt prefix, the naming convention the
    whole HLT menu follows. An EDProducer must declare produces<> in its constructor, so
    the type is read from the module's C++ class name at configuration time; there is no
    runtime discovery to be had.
    """
    isHlt = flavour == "hlt"
    producers = process.producers_()
    return sorted(
        label
        for label in _scheduledModuleNames(process)
        if label in producers
        and producers[label].type_() in tracksterProducerTypes
        and label.startswith("hlt") == isHlt
    )


def tracksterLabelsFromInputFile(process, flavour="offline"):
    """Trackster collections the process READS, discovered by branch type in its first
    input file. A RECO or DQM step takes the HLT tracksters from its input and so has no
    HLT producer to look at. Best effort: a remote input, or one the previous step has
    not written yet, yields nothing and the other sources stand.

    The branch name of an EDM product is
    <friendlyClassName>_<label>_<instance>_<process>, and the friendly name of
    std::vector<ticl::Trackster> is ticlTracksters, so this too is a match on type.
    """
    source = getattr(process, "source", None)
    fileNames = getattr(source, "fileNames", None) if source is not None else None
    if not fileNames:
        return []
    path = str(fileNames[0])
    if "://" in path:
        return []
    import ROOT

    try:
        inputFile = ROOT.TFile.Open(path)
    except OSError:
        return []
    if not inputFile or inputFile.IsZombie():
        return []
    events = inputFile.Get("Events")
    prefix = "ticlTracksters_"
    labels = set()
    if events:
        for branch in events.GetListOfBranches():
            name = branch.GetName()
            if name.startswith(prefix):
                labels.add(name[len(prefix):].split("_")[0])
    inputFile.Close()
    isHlt = flavour == "hlt"
    return sorted(label for label in labels if label.startswith("hlt") == isHlt)


def setTracksterLabelsFromProcess(process):
    """Retarget both trackster lists at what `process` produces or reads, keeping the
    registry fallback for a reconstruction neither source can see.

    MUST be called before truthGraphAssociators_cff is imported: it builds its modules
    and instance labels from these lists at import time.
    """
    for consumer in (
        "SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff",
    ):
        if consumer in sys.modules:
            raise RuntimeError(
                "setTracksterLabelsFromProcess must run before " + consumer + " is imported"
            )
    for pset, flavour in ((truthGraphRecoLabelsPSet, "offline"), (truthGraphHltRecoLabelsPSet, "hlt")):
        discovered = set(tracksterLabelsFromProcess(process, flavour))
        discovered |= set(tracksterLabelsFromInputFile(process, flavour))
        if discovered:
            pset.tracksters = cms.vstring(*sorted(discovered))
    return recoLabels("tracksters"), recoLabels("tracksters", "hlt")


# Branch levels the truth-driven direction asks about, one denominator product per
# level, side by side. Only hit-based domains have levels: a composite object's truth
# target is a vertex, fixed by its resolution instead.
_truthLevels = cms.vstring(
    "stableLegsFromUpstream", "caloBoundary", "stableDecayProducts", "hardProcess",
    # The resonance's visible final state, which needs LevelFlag::Signal on the graph.
    # Stamped at DIGI, so a sample produced before that carries an empty level.
    "reconstructableFromSignal", "underlyingEvent",
    # One root per parton-initiated jet: the hard-scatter legs that are quarks or gluons,
    # each standing for everything downstream of it. No clustering.
    "partonJets",
    # The weakly decaying hadron of each heavy flavour along a chain, the one CMS ghost
    # association names. Separate levels because a B decays to a D, so one combined
    # level would keep only one flavour per chain.
    "bHadrons", "cHadrons",
    # Event-wide visible final state: the reconstructableFromSignal walk seeded from every
    # GEN root, so a pi0 is one object on samples with no resonance to seed from.
    "reconstructableFinalState",
    # One entry per hadronically decaying tau, the last copy of each radiative chain.
    "visibleTau"
)

# The selection preset's seed species, so the signalSeeds product (the _signal
# efficiency denominator) is the preset's signal object itself. A production that
# applies a preset must set this to the SAME pdgIds the preset seeds with, via
# PhysicsTools.TruthInfo.truthGraphSelections.seedPdgIdsForPreset. With no preset
# there is no resonance and the signal products stay empty.
_signalSeedPdgIds = cms.vint32()
_signalSeedHadronFlavors = cms.vint32()

def instanceKey(label):
    """Product instance key for a collection label: label and instance joined by an
    underscore. HGCal uses no separator for product labels but an underscore for DQM
    folder names; this package uses the underscore for BOTH so a key reads the same
    wherever it appears."""
    return label.replace(":", "_")


