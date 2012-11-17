import FWCore.ParameterSet.Config as cms

from Validation.HcalHits.SimHitsValidationHcal_cfi import *
import Validation.HcalHits.SimHitsValidationHcal_cfi
AllSimHitsValidation = Validation.HcalHits.SimHitsValidationHcal_cfi.simHitsValidationHcal.clone()
SimHitsValidationSequence = cms.Sequence(AllSimHitsValidation)
