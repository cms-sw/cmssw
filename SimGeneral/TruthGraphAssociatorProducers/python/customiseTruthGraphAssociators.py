# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Schedule the truth-graph association producers at RECO and keep their products.
# The producers form a cms.Task, so the framework runs only what is actually consumed
# and resolves the constituent dependency (vertices need the track maps) itself.

import FWCore.ParameterSet.Config as cms


def customiseTruthGraphAssociators(process):
    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
        setTracksterLabelsFromProcess,
    )

    # Discover the trackster collections by producer type before importing the
    # associator cff, which builds its modules from the label lists at import time.
    setTracksterLabelsFromProcess(process)

    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import (
        truthBranchTargets,
        allTrackToTruthBranchAssociators,
        allVertexToTruthBranchAssociators,
        allSecondaryVertexToTruthBranchAssociators,
        truthBranchTracksterAssociators,
        hltTrackToTruthBranchAssociators,
        hltVertexToTruthBranchAssociators,
        hltTruthBranchTracksterAssociators,
    )

    # Attach each producer to the process FIRST: a cms.Task imported by name carries
    # modules that have no label yet, and adding it directly fails with "an entry in
    # task ... has not been attached to the process".
    process.truthBranchTargets = truthBranchTargets
    process.allTrackToTruthBranchAssociators = allTrackToTruthBranchAssociators
    process.allVertexToTruthBranchAssociators = allVertexToTruthBranchAssociators
    process.allSecondaryVertexToTruthBranchAssociators = allSecondaryVertexToTruthBranchAssociators
    process.truthBranchTracksterAssociators = truthBranchTracksterAssociators
    process.hltTrackToTruthBranchAssociators = hltTrackToTruthBranchAssociators
    process.hltVertexToTruthBranchAssociators = hltVertexToTruthBranchAssociators
    process.hltTruthBranchTracksterAssociators = hltTruthBranchTracksterAssociators

    # A Sequence, not a Task: a Task runs a module only when another module consumes its
    # product, so a job with no output module and no validation would apply this customise
    # and produce nothing. The order is the data flow: the targets first, then the
    # hit-based domains, then the composite domains, which consume the track maps.
    process.truthGraphAssociatorsSequence = cms.Sequence(
        process.truthBranchTargets
        + process.allTrackToTruthBranchAssociators
        + process.truthBranchTracksterAssociators
        + process.allVertexToTruthBranchAssociators
        + process.allSecondaryVertexToTruthBranchAssociators
        + process.hltTrackToTruthBranchAssociators
        + process.hltTruthBranchTracksterAssociators
        + process.hltVertexToTruthBranchAssociators
    )
    process.truthGraphAssociatorsPath = cms.Path(process.truthGraphAssociatorsSequence)
    if process.schedule is not None:
        process.schedule.append(process.truthGraphAssociatorsPath)

    for out in process.outputModules_().values():
        out.outputCommands.extend(
            [
                "keep *_truthBranchTargets_*_*",
                "keep *_allTrackToTruthBranchAssociators_*_*",
                "keep *_allVertexToTruthBranchAssociators_*_*",
                "keep *_allSecondaryVertexToTruthBranchAssociators_*_*",
                "keep *_truthBranchTracksterAssociators_*_*",
                "keep *_hltTrackToTruthBranchAssociators_*_*",
                "keep *_hltVertexToTruthBranchAssociators_*_*",
                "keep *_hltTruthBranchTracksterAssociators_*_*",
            ]
        )
    return process
