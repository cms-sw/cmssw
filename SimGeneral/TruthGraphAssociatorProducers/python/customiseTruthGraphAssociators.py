# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Schedule the truth-graph association producers at RECO, offline and HLT, and keep
# their products. The release validation schedules the offline sequence on its own,
# without this customise; use this one for a job that wants the maps in its output.

import sys

import FWCore.ParameterSet.Config as cms

_associatorsCff = "SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff"


def customiseTruthGraphAssociators(process):
    # The associator cff builds its modules from the trackster label lists at import
    # time, so the lists can only be retargeted while it is not imported yet. A job with
    # VALIDATION imports it through globalValidation_cff, and there the labels are the
    # ones of the registry.
    if _associatorsCff not in sys.modules:
        from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociationLabels_cff import (
            setTracksterLabelsFromProcess,
        )

        setTracksterLabelsFromProcess(process)

    # load(), not import: it labels every module of the cff on the process, which a
    # sequence imported by name cannot do for itself.
    if not hasattr(process, "truthGraphAssociatorsSequence"):
        process.load(_associatorsCff)

    # The offline sequence is already scheduled in a job that runs the validation. A
    # module in two paths still runs once, and the path carries the HLT twins, which no
    # other sequence schedules.
    if not hasattr(process, "truthGraphAssociatorsPath"):
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
