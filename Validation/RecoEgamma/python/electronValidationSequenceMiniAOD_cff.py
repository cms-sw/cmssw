import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi import * 
from Validation.RecoEgamma.ElectronIsolation_cfi import *

electronValidationSequenceMiniAOD = cms.Sequence(ElectronIsolation + electronMcSignalValidatorMiniAOD)

