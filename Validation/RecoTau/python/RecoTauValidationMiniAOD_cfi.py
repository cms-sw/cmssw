import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

tauValidationMiniAOD = DQMEDAnalyzer("TauValidationMiniAOD",
  tauCollection = cms.InputTag("slimmedTaus"),
)

