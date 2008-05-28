import FWCore.ParameterSet.Config as cms

from Validation.HcalDigis.HcalDigiParam_cfi import *
import Validation.HcalDigis.HcalDigiParam_cfi
HBDigisValidation = Validation.HcalDigis.HcalDigiParam_cfi.hcalDigiAnalyzer.clone()
import Validation.HcalDigis.HcalDigiParam_cfi
HEDigisValidation = Validation.HcalDigis.HcalDigiParam_cfi.hcalDigiAnalyzer.clone()
import Validation.HcalDigis.HcalDigiParam_cfi
HODigisValidation = Validation.HcalDigis.HcalDigiParam_cfi.hcalDigiAnalyzer.clone()
import Validation.HcalDigis.HcalDigiParam_cfi
HFDigisValidation = Validation.HcalDigis.HcalDigiParam_cfi.hcalDigiAnalyzer.clone()
hcalDigisValidationSequence = cms.Sequence(HBDigisValidation+HEDigisValidation+HODigisValidation+HFDigisValidation)
HBDigisValidation.hcalselector = 'HB'
HBDigisValidation.outputFile = 'HcalDigisValidationHB.root'
HEDigisValidation.hcalselector = 'HE'
HEDigisValidation.outputFile = 'HcalDigisValidationHE.root'
HODigisValidation.hcalselector = 'HO'
HODigisValidation.outputFile = 'HcalDigisValidationHO.root'
HFDigisValidation.hcalselector = 'HF'
HFDigisValidation.outputFile = 'HcalDigisValidationHF.root'


