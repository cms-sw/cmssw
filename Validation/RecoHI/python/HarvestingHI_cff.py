import FWCore.ParameterSet.Config as cms

from Validation.Configuration.postValidation_cff import *

postValidationHI = cms.Sequence(recoMuonPostProcessors+postProcessorTrackSequence)
