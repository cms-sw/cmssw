# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Constructs every truth-branch validator with requireHits on. The job reads no event.

import FWCore.ParameterSet.Config as cms
from Validation.TruthInfo.customiseTruthBranchRequireHits import customiseTruthBranchRequireHits

process = cms.Process("TEST")
process.load("DQMServices.Core.DQMStore_cfi")
process.load("SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff")
process.load("Validation.TruthInfo.truthBranchValidation_cff")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 0
process.p = cms.Path(process.truthGraphAssociatorsSequence + process.truthBranchValidationSequence)
process = customiseTruthBranchRequireHits(process)
