# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# The association producers, one per domain, all driven by the label and working-point
# lists in truthGraphAssociationLabels_cff so a collection is configured in exactly one
# place. Composite domains consume the constituent domain's maps, so the order in the
# sequences below matters only for readability: the framework resolves the data
# dependency itself.

import FWCore.ParameterSet.Config as cms

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
    truthBranchWorkingPointsPSet,
    recoLabels,
    _truthLevels,
    _signalSeedPdgIds,
    _signalSeedHadronFlavors,
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

# Which candidate roots a reco object may be ASSIGNED to. A root carries the hits of its
# whole subgraph, so a parton or a beam particle covers the object entirely and would win
# on score alone. The barred roots stay candidates: they are the members of the
# hardProcess and partonJets denominators, which the truth-driven direction must reach.
truthAssignableTargetsBlock = cms.PSet(
    excludeSynthetic=cms.bool(True),
    excludeArtificialProduction=cms.bool(True),
    excludeBeamParticles=cms.bool(True),
    excludePartons=cms.bool(True),
    excludeElectroweakBosons=cms.bool(True),
    extraBarredPdgIds=cms.vint32(),
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
    assignableTargets=truthAssignableTargetsBlock.clone(),
)

# Hit-based: the object owns detector hits.
allTrackToTruthBranchAssociators = cms.EDProducer(
    "AllTrackToTruthBranchAssociatorsProducer",
    recoCollections=_tags("tracks"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    assignableTargetsSrc=cms.InputTag("truthBranchTargets", "assignableRoots"),
    **_truthSources,
    **_workingPointArgs,
)

# Constituent-based: a vertex has no hits of its own, its truth is aggregated from the
# tracks it is built from, which the track associator has already matched.
allVertexToTruthBranchAssociators = cms.EDProducer(
    "AllVertexToTruthBranchAssociatorsProducer",
    recoCollections=_tags("vertices"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    assignableTargetsSrc=cms.InputTag("truthBranchTargets", "assignableRoots"),
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
    assignableTargetsSrc=cms.InputTag("truthBranchTargets", "assignableRoots"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    # A trackster is an endcap object, so its shared-energy denominator covers the HGCAL
    # only, not the whole Calo hit channel. See the producer's fillDescriptions.
    denominatorDetectors=cms.vstring("HGCalEE", "HGCalHSi", "HGCalHSc"),
    **_truthSources,
    **_workingPointArgs,
)

# Hit-based on the CALORIMETER channel, barrel: a particle-flow cluster owns its cells
# directly through hitsAndFractions, so it is matched on shared energy like a
# trackster, with no layer-cluster collection to resolve. One module per subdetector:
# denominatorDetectors is a single mask per module and the shared-energy fraction that
# gates the efficiency is normalised to the branch energy in those detectors, so an
# ECAL cluster must be scored against the branch's ECAL energy, not ECAL plus HCAL.
truthBranchPFClusterEcalAssociators = cms.EDProducer(
    "TruthBranchPFClusterAssociatorsProducer",
    recoCollections=_tags("pfClustersEcal"),
    targetsSrc=cms.InputTag("truthBranchTargets", "selectedRoots"),
    assignableTargetsSrc=cms.InputTag("truthBranchTargets", "assignableRoots"),
    denominatorDetectors=cms.vstring("Ecal"),
    **_truthSources,
    **_workingPointArgs,
)
truthBranchPFClusterHcalAssociators = truthBranchPFClusterEcalAssociators.clone(
    recoCollections=_tags("pfClustersHcal"),
    denominatorDetectors=["Hcal"],
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
    hgcalRecHits=cms.VInputTag(
        cms.InputTag("hltHGCalRecHit", "HGCEERecHits"),
        cms.InputTag("hltHGCalRecHit", "HGCHEFRecHits"),
        cms.InputTag("hltHGCalRecHit", "HGCHEBRecHits"),
    ),
    pfRecHits=cms.VInputTag("hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"),
)

# Sequences, not Tasks: a Task runs a module only when another module consumes its
# product, so a job that keeps the maps without validating them would produce nothing.
# The order is the data flow, the targets first, then the hit-based domains, then the
# composite domains, which consume the track maps.
truthGraphAssociatorsSequence = cms.Sequence(
    truthBranchTargets
    + allTrackToTruthBranchAssociators
    + truthBranchTracksterAssociators
    + truthBranchPFClusterEcalAssociators
    + truthBranchPFClusterHcalAssociators
    + allVertexToTruthBranchAssociators
    + allSecondaryVertexToTruthBranchAssociators
)

# The HLT twins read HLT collections, which an offline reconstruction does not produce,
# so they are kept apart from the offline sequence.
truthGraphHltAssociatorsSequence = cms.Sequence(
    hltTrackToTruthBranchAssociators + hltTruthBranchTracksterAssociators + hltVertexToTruthBranchAssociators
)
