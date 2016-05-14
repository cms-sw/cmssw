import FWCore.ParameterSet.Config as cms

from Validation.Configuration.postValidation_cff import *

# to be customized for OLD or NEW validation
#postValidationHI = cms.Sequence(recoMuonPostProcessors+postProcessorTrackSequence)
postValidationHI = cms.Sequence(NEWrecoMuonPostProcessors+postProcessorTrackSequence)
