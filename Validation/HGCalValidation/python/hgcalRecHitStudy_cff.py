import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitStudyEE_cfi import *

hgcalRecHitStudyFH = hgcalRecHitStudyEE.clone(
    detectorName  = cms.string("HGCalHESiliconSensitive"),
    source  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitStudyBH = hgcalRecHitStudyEE.clone(
    detectorName  = cms.string("HCal"),
    source  = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalRecHitStudyBH,
    detectorName  = cms.string("HGCalHEScintillatorSensitive"),
)
