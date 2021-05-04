import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitValidationEE_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify( hgcalSimHitValidationEE, fromDDD = False )

hgcalSimHitValidationHEF = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"),
    CaloHitSource = cms.string("HGCHitsHEfront"))

hgcalSimHitValidationHEB = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
    CaloHitSource = cms.string("HGCHitsHEback"),
)
