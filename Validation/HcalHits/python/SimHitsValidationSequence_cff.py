import FWCore.ParameterSet.Config as cms

from Validation.HcalHits.SimHitsValidationHcal_cfi import *
import Validation.HcalHits.SimHitsValidationHcal_cfi

from Validation.HcalHits.HcalSimHitsValidation_cfi import *
import Validation.HcalHits.HcalSimHitsValidation_cfi

AllSimHitsValidation = Validation.HcalHits.SimHitsValidationHcal_cfi.simHitsValidationHcal.clone()
HcalSimHitsAnalyser = Validation.HcalHits.HcalSimHitsValidation_cfi.HcalSimHitsAnalyser.clone()

hcalSimHitsValidationSequence = cms.Sequence(AllSimHitsValidation*HcalSimHitsAnalyser)
