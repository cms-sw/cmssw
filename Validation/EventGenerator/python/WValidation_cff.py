import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.WValidation_cfi import *

wMinusEleValidation = wValidation.clone(decaysTo = cms.int32(11), name = cms.string("Electrons"))
wPlusEleValidation = wValidation.clone(decaysTo = cms.int32(-11), name = cms.string("Positrons"))

wMinusMuoValidation = wValidation.clone(decaysTo = cms.int32(13), name = cms.string("Muons"))
wPlusMuoValidation = wValidation.clone(decaysTo = cms.int32(-13), name = cms.string("AntiMuons"))

wEleValidation = cms.Sequence(wMinusEleValidation+wPlusEleValidation)
wMuoValidation = cms.Sequence(wMinusMuoValidation+wPlusMuoValidation)

