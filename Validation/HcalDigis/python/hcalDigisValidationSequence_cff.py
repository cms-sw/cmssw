import FWCore.ParameterSet.Config as cms

from Validation.HcalDigis.HcalDigisParam_cfi import *
import Validation.HcalDigis.HcalDigisParam_cfi
AllHcalDigisValidation = Validation.HcalDigis.HcalDigisParam_cfi.hcaldigisAnalyzer.clone()
hcaldigisValidationSequence = cms.Sequence(AllHcalDigisValidation)



