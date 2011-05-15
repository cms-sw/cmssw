import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *

from TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cfi import scrapingFilter

eventCleaning = cms.Sequence(
  HBHENoiseFilter
+ scrapingFilter
)
