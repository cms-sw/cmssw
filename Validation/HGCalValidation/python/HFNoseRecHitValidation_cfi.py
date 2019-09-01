import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitValidationEE_cfi import *

hfnoseRecHitValidation = hgcalRecHitValidationEE.clone(
    DetectorName = cms.string("HGCalHFNoseSensitive"),
    RecHitSource = cms.InputTag("HGCalRecHit","HGCHFNoseRecHits"))

