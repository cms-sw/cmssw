import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitStudyEE_cfi import *

hgcalRecHitStudyFH = hgcalRecHitStudyEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitStudyBH = hgcalRecHitStudyEE.clone(
    DetectorName  = cms.string("HCal"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))
