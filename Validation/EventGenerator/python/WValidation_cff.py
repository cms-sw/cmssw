import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.WValidation_cfi import *

wMinusEleValidation = wValidation.clone(decaysTo = 11, name = "Electrons")
wPlusEleValidation = wValidation.clone(decaysTo = -11, name = "Positrons")

wMinusMuoValidation = wValidation.clone(decaysTo = 13, name = "Muons")
wPlusMuoValidation = wValidation.clone(decaysTo = -13, name = "AntiMuons")

wEleValidation = cms.Sequence(wMinusEleValidation+wPlusEleValidation)
wMuoValidation = cms.Sequence(wMinusMuoValidation+wPlusMuoValidation)

