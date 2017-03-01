import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonValidatorMiniAOD_cfi import *


photonMiniAODValidationSequence = cms.Sequence(photonValidationMiniAOD)


