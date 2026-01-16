import FWCore.ParameterSet.Config as cms
from Validation.RecoJets.JetPostProcessor_cfi import *

JetPostProcessorHarvesting = cms.Sequence(
    JetTesterPostprocessing
    + JetPostProcessor
)
