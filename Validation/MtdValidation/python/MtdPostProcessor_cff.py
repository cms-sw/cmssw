import FWCore.ParameterSet.Config as cms

from Validation.MtdValidation.btlSimHitsPostProcessor_cfi import btlSimHitsPostProcessor

mtdValidationPostProcessor = cms.Sequence(btlSimHitsPostProcessor)
