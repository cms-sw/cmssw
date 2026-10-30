# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""requireHits of every hit-based validator cuts on the detectors of its associator."""

import unittest

import FWCore.ParameterSet.Config as cms
from Validation.TruthInfo.customiseTruthBranchRequireHits import (
    customiseTruthBranchRequireHits,
    hitBasedValidatorTypes,
)


def makeProcess():
    process = cms.Process("TEST")
    process.load("SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff")
    process.load("Validation.TruthInfo.truthBranchValidation_cff")
    # load() attaches the module objects of the cff, so a test that changes them works on copies.
    for label, module in list(process.producers_().items()):
        setattr(process, label, module.clone())
    return process


def validators(process, types):
    return {label: module for label, module in process.producers_().items() if module.type_() in types}


def associatorDetectors(process, validator):
    associator = getattr(process, validator.associator.value())
    if not associator.hasParameter("denominatorDetectors"):
        return []
    return list(associator.denominatorDetectors.value())


class TestRequireHits(unittest.TestCase):
    def test_defaultValidatorsUseTheDetectorsOfTheirAssociator(self):
        process = makeProcess()
        found = validators(process, hitBasedValidatorTypes)
        self.assertGreaterEqual(len(found), 4)
        for label, module in found.items():
            self.assertFalse(module.hasParameter("requireHits") and module.requireHits.value(), label)
            self.assertEqual(list(module.denominatorDetectors.value()), associatorDetectors(process, module), label)

    def test_customiseTurnsOnEveryHitBasedValidator(self):
        process = customiseTruthBranchRequireHits(makeProcess())
        for label, module in validators(process, hitBasedValidatorTypes).items():
            self.assertTrue(module.requireHits.value(), label)
        for label, module in validators(process, ("TruthBranchVertexValidator",)).items():
            self.assertFalse(module.hasParameter("requireHits"), label)

    def test_customiseFollowsTheAssociatorOfTheProcess(self):
        process = makeProcess()
        process.truthBranchTracksterAssociators.denominatorDetectors = ["HGCalEE"]
        process = customiseTruthBranchRequireHits(process)
        self.assertEqual(list(process.truthBranchTracksterValidator.denominatorDetectors.value()), ["HGCalEE"])

    def test_theTypesAreTheHitBasedModulesOfTheValidationCff(self):
        import Validation.TruthInfo.truthBranchValidation_cff as validation

        types = {d["module"] for d in validation._domains if "vertexResolution" not in d}
        self.assertEqual(types, set(hitBasedValidatorTypes))


if __name__ == "__main__":
    unittest.main()
