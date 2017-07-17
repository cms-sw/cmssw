import FWCore.ParameterSet.Config as cms

from Validation.RecoJets.JetPostProcessor_cfi import *
JetPostProcessor = cms.Sequence(JetPostprocessing)
