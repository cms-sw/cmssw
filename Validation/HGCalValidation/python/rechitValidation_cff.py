import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitValidationEE_cfi import *

hgcalRecHitValidationHEF = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitValidationHEB = hgcalRecHitValidationEE.clone(
    DetectorName  = cms.string("HCal"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify(hgcalRecHitValidationHEB, DetectorName = cms.string("HGCalHEScintillatorSensitive"));

