import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

import Validation.HcalRecHits.HcalRecHitParam_cfi
AllRecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
hcalRecHitsValidationSequence = cms.Sequence(AllRecHitsValidation)

AllRecHitsValidation.hcalselector = 'all'
AllRecHitsValidation.outputFile = ''


