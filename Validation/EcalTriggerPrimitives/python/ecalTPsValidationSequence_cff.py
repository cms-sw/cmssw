import FWCore.ParameterSet.Config as cms

ecalTPsValidationSequence = cms.Sequence()

from Validation.EcalTriggerPrimitives.ecalTPsValidationPh2_cfi import *
ecalTPsValidationSequencePhase2 = cms.Sequence(ecalTPsValidationPh2)
