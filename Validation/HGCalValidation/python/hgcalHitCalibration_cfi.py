import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalHitCalibrationDefault_cfi import hgcalHitCalibrationDefault as _hgcalHitCalibrationDefault
hgcalHitCalibration = _hgcalHitCalibrationDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalHitCalibration, caloParticles = "mixData:MergedCaloTruth")
