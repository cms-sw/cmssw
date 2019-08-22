import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitValidationEE_cfi import *

hfnoseSimHitValidationHEF = hgcalSimHitValidationEE.clone(
    DetectorName  = cms.string("HGCalHFNoseSensitive"),
    CaloHitSource = cms.string("HFNoseHits"))
