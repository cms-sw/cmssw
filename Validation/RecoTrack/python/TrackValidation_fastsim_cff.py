import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
TrackAssociatorByHitsRecoDenom.associateStrip = False
TrackAssociatorByHitsRecoDenom.associatePixel = False
TrackAssociatorByHitsRecoDenom.ROUList = ['famosSimHitsTrackerHits']
multiTrackValidator.UseAssociators = True
multiTrackValidator.skipHistoFit=cms.untracked.bool(True)
multiTrackValidator.useLogPt=cms.untracked.bool(True)
multiTrackValidator.minpT = cms.double(-1)
multiTrackValidator.maxpT = cms.double(3)
multiTrackValidator.nintpT = cms.int32(40)
###must be commented in normal running
###multiTrackValidator.outputFile='validationPlots.root'


