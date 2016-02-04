import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.DrellYanValidation_cfi import *

drellYanEleValidation = drellYanValidation.clone(decaysTo = cms.int32(11), name = cms.string("Electrons"))

drellYanMuoValidation = drellYanValidation.clone(decaysTo = cms.int32(13), name = cms.string("Muons"))
