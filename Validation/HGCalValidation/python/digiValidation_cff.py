import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.digiValidation_cfi import *

hgcalDigiValidationHEF = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HGCalHESiliconSensitive"),
    DigiSource   = cms.InputTag("mix","HGCDigisHEfront"))

hgcalDigiValidationHEB = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HGCalHEScintillatorSensitive"),
    DigiSource   = cms.InputTag("mix","HGCDigisHEback"))
