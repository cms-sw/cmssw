import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.electronValidationSequenceMiniAOD_cff import *

egammaValidationMiniAOD = cms.Sequence( electronValidationSequenceMiniAOD )
