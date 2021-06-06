import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiValidationEE_cfi import *

hgcalDigiValidationHEF = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HGCalHESiliconSensitive"),
    DigiSource   = cms.InputTag("hgcalDigis","HEfront"))

hgcalDigiValidationHEB = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HGCalHEScintillatorSensitive"),
    DigiSource   = cms.InputTag("hgcalDigis","HEback"))
