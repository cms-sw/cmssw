import FWCore.ParameterSet.Config as cms

from Validation.EcalDigis.ecalMixingModuleValidation_cfi import *
ecalMixingModuleValidation.EBdigiCollection = 'ecalUnsuppressedDigis'
ecalMixingModuleValidation.EEdigiCollection = 'ecalUnsuppressedDigis'
ecalMixingModuleValidation.ESdigiCollection = 'ecalUnsuppressedDigis'

