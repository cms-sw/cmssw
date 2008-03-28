import FWCore.ParameterSet.Config as cms

# Run the complete ECAL rechits validation set
from Validation.EcalRecHits.ecalRecHitsValidation_cfi import *
from Validation.EcalRecHits.ecalBarrelRecHitsValidation_cfi import *
from Validation.EcalRecHits.ecalEndcapRecHitsValidation_cfi import *
from Validation.EcalRecHits.ecalPreshowerRecHitsValidation_cfi import *
ecalRecHitsValidationSequence = cms.Sequence(ecalRecHitsValidation*ecalBarrelRecHitsValidation*ecalEndcapRecHitsValidation*ecalPreshowerRecHitsValidation)

