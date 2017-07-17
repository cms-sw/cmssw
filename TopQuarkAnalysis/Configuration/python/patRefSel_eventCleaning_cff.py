import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.goodvertexSkim_cff import noscraping

eventCleaningData = cms.Sequence(
  noscraping
)
eventCleaningMiniAODData = cms.Sequence(
)

eventCleaningMC = cms.Sequence(
)

eventCleaningMiniAODMC = cms.Sequence(
)

#from RecoMET.METFilters.metFilters_cff import * # FIXME: enable after filter sequence has been fixed upstream (e.g. missing 'TobTecFakesFilter')
from RecoMET.METFilters.metFilters_cff import metFilters, HBHENoiseFilter, CSCTightHaloFilter, hcalLaserEventFilter, EcalDeadCellTriggerPrimitiveFilter, goodVertices, trackingFailureFilter, eeBadScFilter, ecalLaserCorrFilter, manystripclus53X, toomanystripclus53X, logErrorTooManyClusters

eventCleaning = cms.Sequence(
  metFilters
)

from TopQuarkAnalysis.Configuration.patRefSel_eventCleaning_cfi import metFiltersMiniAOD

eventCleaningMiniAOD = cms.Sequence(
  metFiltersMiniAOD
)
