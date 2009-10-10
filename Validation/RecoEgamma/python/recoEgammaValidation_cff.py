import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalValidator_cfi import *
from Validation.RecoEgamma.ElectronMcFakeValidator_cfi import *
recoEgammaValidation = cms.Sequence(electronMcSignalValidator+electronMcFakeValidator)
