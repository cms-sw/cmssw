import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessor_cff import *

postValidation = cms.Sequence(postProcessorMuonMultiTrack+postProcessorRecoMuon)
