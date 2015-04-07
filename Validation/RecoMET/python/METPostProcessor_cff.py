import FWCore.ParameterSet.Config as cms

from Validation.RecoMET.METPostProcessor_cfi import *
METPostProcessor = cms.Sequence(METPostprocessing)
METPostProcessorHarvesting = cms.Sequence(METPostprocessingHarvesting)
