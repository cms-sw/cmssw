import FWCore.ParameterSet.Config as cms

from Validation.HGCalHitValidation.hgcalHitCalibration_cfi import hgcalHitCalibration

hgcalHitValidationSequece = cms.Sequence(hgcalHitCalibration)
