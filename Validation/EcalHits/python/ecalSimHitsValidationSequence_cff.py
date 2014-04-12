import FWCore.ParameterSet.Config as cms

# Run the complete ECAL hits validation set
from Validation.EcalHits.ecalSimHitsValidation_cfi import *
from Validation.EcalHits.ecalBarrelSimHitsValidation_cfi import *
from Validation.EcalHits.ecalEndcapSimHitsValidation_cfi import *
from Validation.EcalHits.ecalPreshowerSimHitsValidation_cfi import *
ecalSimHitsValidationSequence = cms.Sequence(ecalSimHitsValidation*ecalBarrelSimHitsValidation*ecalEndcapSimHitsValidation*ecalPreshowerSimHitsValidation)

