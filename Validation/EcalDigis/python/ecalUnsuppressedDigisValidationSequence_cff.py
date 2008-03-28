import FWCore.ParameterSet.Config as cms

# Run the complete ECAL digis validation set
from Validation.EcalDigis.ecalDigisValidation_cfi import *
from Validation.EcalDigis.ecalBarrelDigisValidation_cfi import *
from Validation.EcalDigis.ecalEndcapDigisValidation_cfi import *
from Validation.EcalDigis.ecalPreshowerDigisValidation_cfi import *
ecalUnsuppressedDigisValidationSequence = cms.Sequence(ecalDigisValidation*ecalBarrelDigisValidation*ecalEndcapDigisValidation*ecalPreshowerDigisValidation)
ecalDigisValidation.EBdigiCollection = 'ecalUnsuppressedDigis'
ecalDigisValidation.EEdigiCollection = 'ecalUnsuppressedDigis'
ecalDigisValidation.ESdigiCollection = 'ecalUnsuppressedDigis'
ecalBarrelDigisValidation.EBdigiCollection = 'ecalUnsuppressedDigis'
ecalEndcapDigisValidation.EEdigiCollection = 'ecalUnsuppressedDigis'
ecalPreshowerDigisValidation.ESdigiCollection = 'ecalUnsuppressedDigis'

