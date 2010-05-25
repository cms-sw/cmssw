import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.WValidation_cfi import *

wEleValidation = wValidation.clone(decaysTo = cms.int32(11), name = cms.string("Electrons"))

wMuoValidation = wValidation.clone(decaysTo = cms.int32(13), name = cms.string("Muons"))
