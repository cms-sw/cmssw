import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.electronValidationSequence_cff import *

egammaValidationMiniAOD = cms.Sequence(electronValidationSequenceMiniAOD)
