import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcMiniAODSignalValidator_cfi import * 
 
electronValidationSequenceMiniAOD = cms.Sequence(electronMcMiniAODSignalValidator)

