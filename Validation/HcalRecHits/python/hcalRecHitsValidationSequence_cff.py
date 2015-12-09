import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

import Validation.HcalRecHits.HcalRecHitParam_cfi

RecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
NoiseRatesValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalNoiseRates.clone()

hcalRecHitsValidationSequence = cms.Sequence(NoiseRatesValidation*RecHitsValidation)

# fastsim hasn't got the right noise collection for the moment => no noise validation
from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    hcalRecHitsValidationSequence.remove(NoiseRatesValidation)
