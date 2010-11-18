import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalValidator_cfi import *
from Validation.RecoEgamma.ElectronMcFakeValidator_cfi import *
from Validation.RecoEgamma.photonValidationSequence_cff import *

photonValidation.isRunCentrally = True
tkConversionValidation.isRunCentrally = True

egammaValidation = cms.Sequence(electronMcSignalValidator+electronMcFakeValidator+photonValidationSequence)
