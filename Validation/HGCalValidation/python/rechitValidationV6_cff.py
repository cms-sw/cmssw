import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.rechitValidation_cfi import *

hgcalRecHitValidationHEF = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitValidationHEB = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HCal"),
    ifHCAL        = cms.bool(True),
    RecHitSource  = cms.InputTag("hbhereco"))
