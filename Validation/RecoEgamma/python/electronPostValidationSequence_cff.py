import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalPostValidator_cfi import *
from Validation.RecoEgamma.ElectronMcFakePostValidator_cfi import *
from Validation.RecoEgamma.ElectronMcSignalPostValidatorPt1000_cfi import *

electronPostValidationSequence = cms.Sequence(electronMcSignalPostValidator+electronMcFakePostValidator+electronMcSignalPostValidatorPt1000)
# foo bar baz
# ce4XIc9Dfu4xb
# 5JvrwAxh3GApb
