import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiValidationEE_cfi import *

hfnoseDigiValidation = hgcalDigiValidationEE.clone(
    DetectorName = cms.string("HFNoseSensitive"),
    DigiSource   = cms.InputTag("hgcalDigis","HFNoseDigis"))

