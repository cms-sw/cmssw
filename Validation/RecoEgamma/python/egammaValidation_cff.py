import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalValidator_cfi import *
from Validation.RecoEgamma.ElectronMcFakeValidator_cfi import *
from Validation.RecoEgamma.photonValidationSequence_cff import *

photonValidation.Verbosity = cms.untracked.int32(0)
photonValidation.OutputMEsInRootFile = True


egammaValidation = cms.Sequence(electronMcSignalValidator+electronMcFakeValidator+photonValidationSequence)
