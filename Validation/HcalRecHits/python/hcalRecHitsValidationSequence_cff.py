import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

import Validation.HcalRecHits.HcalRecHitParam_cfi

RecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
NoiseRatesValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalNoiseRates.clone()

hcalRecHitsValidationSequence = cms.Sequence(NoiseRatesValidation*RecHitsValidation)

# fastsim hasn't got the right noise collection for the moment => no noise validation
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    hcalRecHitsValidationSequence.remove(NoiseRatesValidation)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
_phase2_hcalRecHitsValidationSequence = hcalRecHitsValidationSequence.copyAndExclude([NoiseRatesValidation])
phase2_hcal.toReplaceWith(hcalRecHitsValidationSequence, _phase2_hcalRecHitsValidationSequence)
