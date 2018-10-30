import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitValidationEE_cfi import *

hgcalSimHitValidationHEF = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    CaloHitSource = cms.string("HGCHitsHEfront"))

hgcalSimHitValidationHEB = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HCal"),
    CaloHitSource = cms.string("HcalHits"))


from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalSimHitValidationHEB,
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
    CaloHitSource = cms.string("HGCHitsHEback"),
)
