import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.DrellYanValidation_cfi import *

drellYanEleValidation = drellYanValidation.clone(decaysTo = 11, name = "Electrons")

drellYanMuoValidation = drellYanValidation.clone(decaysTo = 13, name = "Muons")
