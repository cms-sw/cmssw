# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Keep in the truth-to-reco denominators only the truth objects with a hit in the
# detectors that the validated collection covers.

import FWCore.ParameterSet.Config as cms

# The validators whose truth objects are particle branches with hits.
hitBasedValidatorTypes = (
    "TruthBranchTrackValidator",
    "TruthBranchTracksterValidator",
    "TruthBranchPFClusterValidator",
)


def customiseTruthBranchRequireHits(process):
    """Turn on requireHits on every hit-based truth-branch validator of the process.

    Each validator takes the denominatorDetectors of its associator in the process, and the
    customise raises when that associator is not in the process. Apply this customise after
    the one that loads the validators. A harvesting job fills no denominator, so the
    customise changes nothing there.

        cmsDriver.py ... --customise SimGeneral/TruthGraphAssociatorProducers/\
            customiseTruthGraphAssociators.customiseTruthBranchValidation,\
            Validation/TruthInfo/customiseTruthBranchRequireHits.customiseTruthBranchRequireHits
    """
    if hasattr(process, "dqmSaver"):
        return process
    for name, module in process.producers_().items():
        if module.type_() not in hitBasedValidatorTypes:
            continue
        module.requireHits = cms.bool(True)
        label = module.associator.value()
        associator = getattr(process, label, None)
        if associator is None:
            raise RuntimeError("requireHits on %s needs the detectors of its associator %s, which is not in the "
                               "process." % (name, label))
        # An associator with no denominatorDetectors keeps the whole hit channel.
        if associator.hasParameter("denominatorDetectors"):
            module.denominatorDetectors = cms.vstring(*associator.denominatorDetectors.value())
    return process
