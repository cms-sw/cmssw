import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

import Validation.HcalRecHits.HcalRecHitParam_cfi

RecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
NoiseRatesValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalNoiseRates.clone()

hcalRecHitsValidationSequence = cms.Sequence(NoiseRatesValidation*RecHitsValidation)

# fastsim hasn't got the right noise collection for the moment => no noise validation
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(hcalRecHitsValidationSequence, hcalRecHitsValidationSequence.copyAndExclude([NoiseRatesValidation]))

_run3_hcalRecHitsValidationSequence = hcalRecHitsValidationSequence.copyAndExclude([NoiseRatesValidation])
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith(hcalRecHitsValidationSequence, _run3_hcalRecHitsValidationSequence)
