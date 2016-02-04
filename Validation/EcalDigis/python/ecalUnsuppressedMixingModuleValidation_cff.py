import FWCore.ParameterSet.Config as cms

from Validation.EcalDigis.ecalMixingModuleValidation_cfi import *
ecalMixingModuleValidation.EBdigiCollection = 'simEcalUnsuppressedDigis'
ecalMixingModuleValidation.EEdigiCollection = 'simEcalUnsuppressedDigis'
ecalMixingModuleValidation.ESdigiCollection = 'simEcalUnsuppressedDigis'

