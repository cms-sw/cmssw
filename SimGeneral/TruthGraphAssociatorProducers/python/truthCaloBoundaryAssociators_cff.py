# The MC-truth graph as a TRAINING truth: the calo-boundary level as the truth objects, and
# the tracksters and layer clusters associated to exactly those objects.
#
# truth::Level::CaloBoundary is one particle per outward crossing of the tracker-calorimeter
# boundary, the particle a SimCluster is built from, with its Geant4 descendants' hits as
# its footprint. Restricting the associators' candidate roots to that level makes the
# "Fixed" map the full one-to-many list of boundary particles sharing energy with a reco
# object (best first) and the adaptive points its single best one; no ancestor is a
# candidate, so nothing can climb above the boundary particle. A layer cluster and a
# trackster are matched against the same list, so a training that starts from either
# input has the same target.
#
# Unlike the validation targets, these include the in-time pile-up particles (the
# associated truth object of a pile-up trackster is then a pile-up particle, flagged as
# such in the particle table) and apply no pt floor.
#
# Reads the graph and the hit index the DIGI step built (truthLogicalGraphProducer,
# truthLogicalGraphHitIndexProducer, kept in the GSD file); it never rebuilds them.

import FWCore.ParameterSet.Config as cms

from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import (
    truthBranchTargets,
    truthBranchTracksterAssociators,
    truthBranchLayerClusterAssociators,
)

truthBranchTargetsCaloBoundary = truthBranchTargets.clone(
    truthLevels=["caloBoundary"],
    # Pile-up boundary particles are targets too; the particle table carries isSignal.
    truthToRecoSignalOnly=False,
    # No pt floor: a soft boundary particle is still the truth of the cells it left.
    branchSelector=dict(ptMin=0.0),
)

_caloBoundaryTargets = cms.InputTag("truthBranchTargetsCaloBoundary", "truthToRecoTargetsCaloBoundary")

truthCaloBoundaryTracksterAssociators = truthBranchTracksterAssociators.clone(
    recoCollections=[cms.InputTag("ticlTrackstersCLUE3DHigh")],
    targetsSrc=_caloBoundaryTargets,
    assignableTargetsSrc=_caloBoundaryTargets,
    # HGCAL only: no barrel PFRecHits, which a RECO output need not carry.
    pfRecHits=cms.VInputTag(),
)

truthCaloBoundaryLayerClusterAssociators = truthBranchLayerClusterAssociators.clone(
    targetsSrc=_caloBoundaryTargets,
    assignableTargetsSrc=_caloBoundaryTargets,
)

# A Sequence, not a Task: the maps are read by the nano step, not by this job.
truthCaloBoundaryAssociatorsSequence = cms.Sequence(
    truthBranchTargetsCaloBoundary
    + truthCaloBoundaryTracksterAssociators
    + truthCaloBoundaryLayerClusterAssociators
)

truthCaloBoundaryKeeps = [
    "keep *_truthBranchTargetsCaloBoundary_*_*",
    "keep *_truthCaloBoundaryTracksterAssociators_*_*",
    "keep *_truthCaloBoundaryLayerClusterAssociators_*_*",
]
