import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitValidationEE_cfi import *

hgcalSimHitValidationHEF = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    CaloHitSource = cms.string("HGCHitsHEfront"))

hgcalSimHitValidationHEB = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
    CaloHitSource = cms.string("HGCHitsHEback"),
)
