import FWCore.ParameterSet.Config as cms
from Validation.HGCalValidation.hgcalDigiValidationEEDefault_cfi import hgcalDigiValidationEEDefault as _hgcalDigiValidationEEDefault
hgcalDigiValidationEE = _hgcalDigiValidationEEDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalDigiValidationEE, DigiSource = "mixData:HGCDigisEE")
