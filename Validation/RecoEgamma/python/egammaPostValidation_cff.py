import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonPostProcessor_cff import *
from Validation.RecoEgamma.electronPostValidationSequence_cff import *

egammaPostValidation = cms.Sequence(photonPostProcessor+electronPostValidationSequence)
# foo bar baz
# BBitqjmSRd8as
# 20OWiDOIqrgaq
