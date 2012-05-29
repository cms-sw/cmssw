import FWCore.ParameterSet.Config as cms

from Validation.HcalDigis.HcalDigisParam_cfi import *
import Validation.HcalDigis.HcalDigisParam_cfi
AllHcalDigisValidation = Validation.HcalDigis.HcalDigisParam_cfi.hcaldigisAnalyzer.clone()
hcaldigisValidationSequence = cms.Sequence(AllHcalDigisValidation)

# the folowing one is a twin of the above and is kept for back compatibility 
# with some old Validation/Configuration/python  sequences... 
hcalDigisValidationSequence = cms.Sequence(AllHcalDigisValidation)



