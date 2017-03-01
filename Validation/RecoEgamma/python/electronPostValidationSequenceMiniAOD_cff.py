import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalPostValidatorMiniAOD_cfi import *

electronPostValidationSequenceMiniAOD = cms.Sequence(electronMcSignalPostValidatorMiniAOD)

