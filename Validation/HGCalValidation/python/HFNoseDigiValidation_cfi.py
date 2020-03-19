import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiValidationEE_cfi import *

hfnoseDigiValidation = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HGCalHFNoseSensitive"),
    DigiSource   = cms.InputTag("hgcalDigis","HFNoseDigis"))

