import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi import * 
 
electronValidationSequenceMiniAOD = cms.Sequence(electronMcSignalValidatorMiniAOD)

