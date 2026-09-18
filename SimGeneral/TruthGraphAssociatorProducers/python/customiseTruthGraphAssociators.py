# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Schedule the reco-to-truth association producers and their DQM, which no sequence
# runs by default. customiseTruthGraphAssociators keeps the maps in the output of a job
# that wants to read them; customiseTruthBranchValidation and customiseTruthHltValidation
# add the performance plots for the offline and the HLT reconstruction.

import sys

import FWCore.ParameterSet.Config as cms

_associatorsCff = "SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff"
_validationCff = "Validation.TruthInfo.truthBranchValidation_cff"


def _loadAssociators(process):
    """Label the association producers on the process."""
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


def _isHarvesting(process):
    """A harvesting job has the DQM saver and no reconstruction to analyse."""
    return hasattr(process, "dqmSaver")


def _hasOfflineReco(process):
    """Whether the process reconstructs what the offline analysers read.

    Every offline domain needs generalTracks, directly or through the vertices, so a
    GEN-SIM, a DIGI or an ALCA job has nothing for them to validate. The offline
    customise is then a no-op, which is what lets one --command carry it through a whole
    relval workflow.
    """
    return hasattr(process, "generalTracks")


def _appendPath(process, name, sequence):
    """Schedule one sequence as its own Path, once and only if it holds modules."""
    # Sum of two sequences: adding them gives a collection, which carries no moduleNames.
    sequence = cms.Sequence(sequence)
    if not sequence.moduleNames():
        return
    if hasattr(process, name):
        return
    setattr(process, name, cms.Path(sequence))
    if process.schedule is not None:
        process.schedule.append(getattr(process, name))


def customiseTruthGraphAssociators(process):
    """Schedule the association producers, offline and HLT, and keep their products.

    Use this for a job that wants the maps in its output file.

        cmsDriver.py ... --customise SimGeneral/TruthGraphAssociatorProducers/\
            customiseTruthGraphAssociators.customiseTruthGraphAssociators
    """
    _loadAssociators(process)
    _appendPath(process, "truthGraphAssociatorsPath",
                process.truthGraphAssociatorsSequence + process.truthGraphHltAssociatorsSequence)

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


def customiseTruthBranchValidation(process):
    """Schedule the offline association maps and their performance plots.

    The plots are the efficiency, the fake rate, the purity and the resolution of every
    reconstructed collection against every truth level and working point. They book 54242
    monitor elements, 21.9 MiB of the harvested DQM file on 10 ttbar D127 events, so only
    a study of the association turns them on. The graph summary and the comparison with
    CaloParticle, SimCluster and TrackingParticle run without this customise, and book 98.

    It books and fills the plots on the reconstruction step and turns them into ratios on
    the harvesting step. Every other step leaves it a no-op, so one option carries it
    through a whole workflow.

        cmsDriver.py ... --customise SimGeneral/TruthGraphAssociatorProducers/\
            customiseTruthGraphAssociators.customiseTruthBranchValidation

        runTheMatrix.py -w upgrade -l 37634.0 --command "--customise SimGeneral/\
            TruthGraphAssociatorProducers/customiseTruthGraphAssociators.\
            customiseTruthBranchValidation"

    A selection preset has to be applied after this customise, so that it reaches the
    analysers and they book their signal folders. --customise_commands always runs last,
    which is the documented way to apply a preset.
    """
    if _isHarvesting(process):
        process.load(_validationCff)
        _appendPath(process, "truthBranchHarvestingPath",
                    process.truthBranchHarvestingSequence)
        return process

    if not _hasOfflineReco(process):
        return process

    # Associators first: the validation cff resolves the reco collection labels when it is
    # imported, and loading the associators is what can retarget them.
    _loadAssociators(process)
    process.load(_validationCff)
    # The analysers read the maps, so the producers share their Path and come first.
    _appendPath(process, "truthBranchValidationPath",
                process.truthGraphAssociatorsSequence + process.truthBranchValidationSequence)
    return process


def customiseTruthHltValidation(process):
    """Schedule the HLT association maps and their performance plots.

    A job has HLT reconstruction to validate only when it ran the HLT menu, so these are
    separate from the offline twins of customiseTruthBranchValidation. Apply this to the
    reconstruction step to book and fill them, and to the harvesting step to turn them
    into ratios.

        cmsDriver.py ... --customise SimGeneral/TruthGraphAssociatorProducers/\
            customiseTruthGraphAssociators.customiseTruthHltValidation
    """
    if _isHarvesting(process):
        process.load(_validationCff)
        _appendPath(process, "truthBranchHltHarvestingPath",
                    process.truthBranchHltHarvestingSequence)
        return process

    _loadAssociators(process)
    process.load(_validationCff)
    _appendPath(process, "truthBranchHltValidationPath",
                process.truthGraphHltAssociatorsSequence + process.truthBranchHltValidationSequence)
    return process
