import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.RecoTauValidationMiniAOD_cfi import *

tauValidationSequenceMiniAOD = cms.Sequence(tauValidationMiniAOD)
