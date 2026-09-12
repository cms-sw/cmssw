# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The association producers, one per domain, all driven by the label and working-point
# lists in truthGraphAssociationLabels_cff so a collection is configured in exactly one
# place. Composite domains consume the constituent domain's maps, so the order in
# truthGraphAssociatorsTask matters only for readability: the framework resolves the
# data dependency itself.

import FWCore.ParameterSet.Config as cms

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
)

# Shared selection of which truth branches are candidates at all. A 1 GeV floor keeps
# the maps and the efficiency denominators from being dominated by soft particles that
# no reconstruction was going to find; loosen it per domain if a study needs to.
truthBranchSelectorBlock = cms.PSet(
    ptMin=cms.float(1.0),
    # ptMax deliberately not set: the producer default is the float maximum, no cut.
    etaMin=cms.float(-4.0),
    etaMax=cms.float(4.0),
    pdgIds=cms.vint32(),
    signalOnly=cms.bool(False),
    intimeOnly=cms.bool(False),
    chargedOnly=cms.bool(False),
    invertEta=cms.bool(False),
    kinematicsOnStableOnly=cms.bool(True),
)

_workingPointArgs = dict(
    workingPointNames=cms.vstring(*truthBranchWorkingPointsPSet.names),
    adaptiveReverseWeight=cms.vfloat(*truthBranchWorkingPointsPSet.adaptiveReverseWeight),
    adaptiveMaxReverseScore=cms.vfloat(*truthBranchWorkingPointsPSet.adaptiveMaxReverseScore),
)

_truthSources = dict(
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
)


def _tags(domain, flavour="offline"):
    return cms.VInputTag(*[cms.InputTag(*label.split(":")) for label in recoLabels(domain, flavour)])


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

# The truth-side targets, once per event: the selector-passing candidate roots the
# associators consume, the signal-seed denominators and one TruthToReco denominator
# per level with its eligibility mask. They depend only on the graph and this
# configuration, so every associator below shares this one producer.
truthBranchTargets = cms.EDProducer(
    "TruthBranchTargetsProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    branchSelector=truthBranchSelectorBlock.clone(),
    truthLevels=_truthLevels,
    signalSeedPdgIds=_signalSeedPdgIds,
    signalSeedHadronFlavors=_signalSeedHadronFlavors,
    truthToRecoSignalOnly=cms.bool(True),
)

# Hit-based: the object owns detector hits.
allTrackToTruthBranchAssociators = cms.EDProducer(
    "AllTrackToTruthBranchAssociatorsProducer",
    recoCollections=_tags("tracks"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    **_truthSources,
    **_workingPointArgs,
)

# Constituent-based: a vertex has no hits of its own, its truth is aggregated from the
# tracks it is built from, which the track associator has already matched.
allVertexToTruthBranchAssociators = cms.EDProducer(
    "AllVertexToTruthBranchAssociatorsProducer",
    recoCollections=_tags("vertices"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    constituentAssociator=cms.string("allTrackToTruthBranchAssociators"),
    constituentCollection=cms.string("generalTracks"),
    # A primary vertex asks which INTERACTION a track came from, so a track produced in
    # a decay downstream of the vertex still counts at the vertex its chain started
    # from. Counting it at its own production vertex would call the whole decay chain of
    # the event contamination.
    vertexResolution=cms.string("interaction"),
    **_truthSources,
    **_workingPointArgs,
)

allSecondaryVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators.clone(
    recoCollections=_tags("secondaryVertices"),
    # A secondary vertex IS a decay or interaction vertex, so the tracks that belong to
    # it were produced there and the immediate production vertex is the right target.
    vertexResolution="immediate",
    # inclusiveSecondaryVertices reconstructs displaced heavy-flavour vertices, so the
    # denominator is those and not every nuclear interaction and conversion in the tracker.
    heavyFlavorOnly=cms.bool(True),
)

# Hit-based on the CALORIMETER channel: a trackster owns energy through its layer
# clusters, so it is matched on shared energy, the same quantity the TICL trackster
# validation scores against.
truthBranchTracksterAssociators = cms.EDProducer(
    "TruthBranchTracksterAssociatorsProducer",
    recoCollections=_tags("tracksters"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    # A trackster is an endcap object, so its shared-energy denominator covers the HGCAL
    # only, not the whole Calo hit channel. See the producer's fillDescriptions.
    denominatorDetectors=cms.vstring("HGCalEE", "HGCalHSi", "HGCalHSc"),
    **_truthSources,
    **_workingPointArgs,
)

# The HLT menu's own reconstruction of the same event. Same producers, same working
# points, different input collections, so the two can be compared page by page.
hltTrackToTruthBranchAssociators = allTrackToTruthBranchAssociators.clone(
    recoCollections=_tags("tracks", "hlt"),
)
hltVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators.clone(
    recoCollections=_tags("vertices", "hlt"),
    constituentAssociator="hltTrackToTruthBranchAssociators",
    constituentCollection="hltGeneralTracks",
)
hltTruthBranchTracksterAssociators = truthBranchTracksterAssociators.clone(
    recoCollections=_tags("tracksters", "hlt"),
    layerClusters="hltMergeLayerClusters",
)

truthGraphAssociatorsTask = cms.Task(
    truthBranchTargets,
    allTrackToTruthBranchAssociators,
    allVertexToTruthBranchAssociators,
    allSecondaryVertexToTruthBranchAssociators,
    truthBranchTracksterAssociators,
    hltTrackToTruthBranchAssociators,
    hltVertexToTruthBranchAssociators,
    hltTruthBranchTracksterAssociators,
)
