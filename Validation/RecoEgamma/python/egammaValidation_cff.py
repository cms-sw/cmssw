import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.electronValidationSequence_cff import *
from Validation.RecoEgamma.photonValidationSequence_cff import *

photonValidation.isRunCentrally = True
pfPhotonValidation.isRunCentrally = True
oldpfPhotonValidation.isRunCentrally = True
tkConversionValidation.isRunCentrally = True

egammaValidation = cms.Sequence(electronValidationSequence+photonValidationSequence)
