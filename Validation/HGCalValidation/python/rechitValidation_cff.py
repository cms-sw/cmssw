import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitValidationEE_cfi import *

hgcalRecHitValidationHEF = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitValidationHEB = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))
# foo bar baz
# 1JeZBgl6aIIad
# L9PZ5WPVqi0CA
