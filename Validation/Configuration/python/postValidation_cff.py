import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessor_cff import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *

postValidation = cms.Sequence(recoMuonPostProcessors+postProcessorTrack)

postValidation_pu = cms.Sequence(postProcessorTrack)
