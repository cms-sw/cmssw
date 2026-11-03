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

    Each validator takes the denominatorDetectors of its associator in the process. Apply
    this customise after the one that loads the validators.

        cmsDriver.py ... --customise SimGeneral/TruthGraphAssociatorProducers/\
            customiseTruthGraphAssociators.customiseTruthBranchValidation,\
            Validation/TruthInfo/customiseTruthBranchRequireHits.customiseTruthBranchRequireHits
    """
    for module in process.producers_().values():
        if module.type_() not in hitBasedValidatorTypes:
            continue
        module.requireHits = cms.bool(True)
        label = module.associator.value() if module.hasParameter("associator") else ""
        associator = getattr(process, label, None)
        if associator is not None and associator.hasParameter("denominatorDetectors"):
            module.denominatorDetectors = cms.vstring(*associator.denominatorDetectors.value())
    return process
