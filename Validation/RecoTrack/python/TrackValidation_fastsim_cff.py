import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
trackAssociatorByHitsRecoDenom.associateStrip = False
trackAssociatorByHitsRecoDenom.associatePixel = False
#TrackAssociatorByHitsRecoDenom.ROUList = ['famosSimHitsTrackerHits']
trackValidator.UseAssociators = True
trackValidator.skipHistoFit=cms.untracked.bool(True)
trackValidator.useLogPt=cms.untracked.bool(True)

trackValidator.stableOnlyTP = True
trackValidator.histoProducerAlgoBlock.generalTpSelector.stableOnly = True
trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.stableOnly = True
trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.stableOnly = True
trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.stableOnly = True
trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.stableOnly = True
trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.stableOnly = True

###must be commented in normal running
###multiTrackValidator.outputFile='validationPlots.root'


