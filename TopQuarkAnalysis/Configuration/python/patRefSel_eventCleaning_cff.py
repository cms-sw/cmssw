import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *
# s. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1196.html
HBHENoiseFilter.minIsolatedNoiseSumE        = 999999.
HBHENoiseFilter.minNumIsolatedNoiseChannels = 999999
HBHENoiseFilter.minIsolatedNoiseSumEt       = 999999.

from RecoMET.METAnalyzers.CSCHaloFilter_cfi import *

from RecoMET.METFilters.hcalLaserEventFilter_cfi import *
hcalLaserEventFilter.vetoByRunEventNumber = cms.untracked.bool( False )
hcalLaserEventFilter.vetoByHBHEOccupancy = cms.untracked.bool( True )

from RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi import *
EcalDeadCellTriggerPrimitiveFilter.tpDigiCollection = cms.InputTag( 'ecalTPSkimNA' )

from RecoMET.METFilters.eeBadScFilter_cfi import *

from RecoMET.METFilters.trackingFailureFilter_cfi import *

from TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cfi import scrapingFilter

eventCleaningData = cms.Sequence(
  scrapingFilter
)

eventCleaningMC = cms.Sequence(
)

eventCleaning = cms.Sequence(
  HBHENoiseFilter
+ CSCTightHaloFilter
+ hcalLaserEventFilter
+ EcalDeadCellTriggerPrimitiveFilter
+ eeBadScFilter
+ trackingFailureFilter
)
