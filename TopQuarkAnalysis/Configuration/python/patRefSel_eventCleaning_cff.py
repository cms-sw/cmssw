import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *
# s. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1196.html
HBHENoiseFilter.minIsolatedNoiseSumE        = 999999.
HBHENoiseFilter.minNumIsolatedNoiseChannels = 999999
HBHENoiseFilter.minIsolatedNoiseSumEt       = 999999.

from TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cfi import scrapingFilter

eventCleaningData = cms.Sequence(
  HBHENoiseFilter
+ scrapingFilter
)
