import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitStudyEE_cfi import *

hgcalRecHitStudyFH = hgcalRecHitStudyEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"))

hgcalRecHitStudyBH = hgcalRecHitStudyEE.clone(
    DetectorName  = cms.string("HCal"),
    RecHitSource  = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalRecHitStudyBH,
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
)
