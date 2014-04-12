import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

import Validation.HcalRecHits.HcalRecHitParam_cfi

RecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
NoiseRatesValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalNoiseRates.clone()

hcalRecHitsValidationSequence = cms.Sequence(NoiseRatesValidation*RecHitsValidation)
