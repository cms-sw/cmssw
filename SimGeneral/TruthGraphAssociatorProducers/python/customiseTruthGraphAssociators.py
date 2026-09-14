# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Schedule the truth-graph association producers at RECO, offline and HLT, and keep
# their products. The release validation schedules the offline sequence on its own,
# without this customise; use this one for a job that wants the maps in its output.

import FWCore.ParameterSet.Config as cms


def customiseTruthGraphAssociators(process):
    from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
        setTracksterLabelsFromProcess,
    )

    # Discover the trackster collections by producer type before importing the
    # associator cff, which builds its modules from the label lists at import time.
    setTracksterLabelsFromProcess(process)

    # load(), not import: it labels every module of the cff on the process, which a
    # sequence imported by name cannot do for itself.
    process.load("SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff")

    process.truthGraphAssociatorsPath = cms.Path(process.truthGraphAssociatorsSequence +
                                                process.truthGraphHltAssociatorsSequence)
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
                "keep *_truthBranchPFClusterEcalAssociators_*_*",
                "keep *_truthBranchPFClusterHcalAssociators_*_*",
                "keep *_hltTrackToTruthBranchAssociators_*_*",
                "keep *_hltVertexToTruthBranchAssociators_*_*",
                "keep *_hltTruthBranchTracksterAssociators_*_*",
            ]
        )
    return process
